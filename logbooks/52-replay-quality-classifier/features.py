"""Structural document features shared by training and inference."""
from dataclasses import asdict, fields
import math
import re
from collections import Counter

import numpy as np

from baseline_rules import ChunkFeatures, chunk_features, chunks, junk_verdict

TRAJECTORY=re.compile(r'This trajectory |# Repository Context|<tool_call>|<\|(?:im_start|start_header_id)\|>|"(?:gold_patch|model_patch|instance_id|test_patch)"')
BENCHMARK=re.compile(r'\b(?:SWE-bench|HumanEval|Terminal-Bench|pass@\d+|benchmark_results)\b',re.I)
QA=re.compile(r'(?:^|\n)\s*(?:Question|Answer|Problem|Solution|Q|A|Pregunta|Respuesta|Frage|Antwort|質問|答え|问题|答案)\s*[:：]',re.I)
HTML=re.compile(r'<(?:html|head|body|script|style|div|table|tr|td|th|span|!DOCTYPE)\b',re.I)
STRUCTURED=re.compile(r'\b(?:INSERT INTO|CREATE TABLE|sourceMappingURL|lockfileVersion|dependencies|integrity)\b')
SPECIAL=re.compile(r'<\|(?:begin_of_text|end_of_text)\|>')
EXTRA_NAMES=('char_entropy','class_entropy','line_p90','line_max','word_mean','trajectory_rate','benchmark_rate','qa_rate','html_rate','structure_rate','replacement_share','ascii_share')
CHUNK_NAMES=tuple(field.name for field in fields(ChunkFeatures))+EXTRA_NAMES
FEATURE_NAMES=('log_tokens','log_visible_chars')+tuple(f'{stat}_{name}' for stat in ('mean','max','min','std') for name in CHUNK_NAMES)


def entropy(counts):
    total=sum(counts)
    return -sum((count/total)*math.log2(count/total) for count in counts if count) if total else 0.0


def measurements(text):
    base=list(asdict(chunk_features(text)).values())
    size=len(text)
    lines=[len(line) for line in text.splitlines()]
    words=text.split()
    classes=Counter('L' if c.isalpha() else 'D' if c.isdigit() else 'S' if c.isspace() else 'P' for c in text)
    extra=[entropy(Counter(text).values()),entropy(classes.values()),np.percentile(lines,90) if lines else 0,max(lines,default=0),
           np.mean([len(word) for word in words]) if words else 0,
           *(len(pattern.findall(text))*1000/size for pattern in (TRAJECTORY,BENCHMARK,QA,HTML,STRUCTURED)),
           text.count('\ufffd')/size,sum(ord(c)<128 for c in text)/size]
    return base+extra


def document_features(windows, tokens):
    cleaned=[SPECIAL.sub('',window) for window in windows]
    rows=[measurements(chunk) for chunk in chunks(cleaned)]
    # Include short documents instead of manufacturing all-zero structural signals.
    if not rows:
        text='\n'.join(cleaned)
        rows=[measurements(text)] if text else [[0.0]*len(CHUNK_NAMES)]
    values=np.asarray(rows,dtype=np.float64)
    features=np.concatenate(([np.log1p(tokens),np.log1p(sum(map(len,cleaned)))],values.mean(axis=0),values.max(axis=0),values.min(axis=0),values.std(axis=0)))
    assert len(features)==len(FEATURE_NAMES)
    return features


def rules_score(windows):
    verdict=junk_verdict([SPECIAL.sub('',window) for window in windows])
    # Continuous junk-chunk share supports calibration at the same false-drop budgets.
    return verdict.junk_share
