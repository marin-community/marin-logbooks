---
title: "Encoder decontamination: a cheaper, higher-recall stage 1"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/81"
related_issues: ["https://github.com/marin-community/marin/pull/11993"]
research_branch: "codex/81-decon-stage1-keys"
marin_base_revision: "2922eb3d0d9d6b00199d027cf62957ca4ef65237"
archive_template_revision: "fc1cb30da0967f6ab78ffd47444ed9144f50cba9"
status: active
---

# Encoder decontamination: a cheaper, higher-recall stage 1

## Question and decision criteria

- Goal and why it matters: stage 1 of the encoder decontamination
  ([marin#11993](https://github.com/marin-community/marin/pull/11993)) scans every catalog record (1.9e10) for
  each eval item's key and sends hits to an encoder classifier (stage 2). A copy without a candidate is never
  scored, so stage 1 sets the recall ceiling. Find which forms of contamination it misses (encodings, edits,
  layouts), how often they occur in real data, and whether a cheap, simple change recovers them.
- Baseline and alternatives: the production keys (below) against signature-gated decodings, a second key, key
  sets matched k of n, word-level TF-IDF/BM25, MinHash/LSH, embedding retrieval, and a dense half-of-windows rule.
- Evidence needed to choose: planted recall by form, real copies found only by the alternative (read and scored
  by stage 2), and the candidate pairs it adds.
- Constraints and unresolved questions: change stage 1 only where a measured gap holds real copies, the simpler
  fix first. Paraphrases and translations are outside any text-overlap stage 1.

## Baseline

- Date: 2026-11-24
- Published source revision: `2922eb3d0d9d6b00199d027cf62957ca4ef65237` (marin#11993 head), keys `stage1-v11`.
- Configuration: the text stream is lowercase letters and digits after NFKC, with LaTeX commands removed. Each item
  has up to three key streams (question; unlettered question with choices for a short question; passage). Key 1
  is the rarest 20-character window of 16 sampled windows, counted over 2 reference shards per source, outside
  the eval's shared prefix and suffix. Readings: as is, JSON unescaped once and twice.
- Result and evidence: planted recall 95.98% over 14 layouts (41,151 items, 827 evals); 98.1% of the block
  rule's clear catalog copies removed at cutoff 0.6 (marin#11993 README).
- Limitations: planted layouts change the layout, not the text; clear copies are exact copies.

## Entries

Unless an entry says otherwise, analyses ran from the base revision with the survey modules in this commit
(`experiments/datakit/decontam/encoder/{encoding,dense,key_set}_survey.py`), on Iris `cw-us-east-02a` at
interactive priority (one single-task job per part, `--part K --parts N`), and from the scripts in
`scripts/` (paths relative to a local `work/` directory). Survey outputs were written to 7-day scratch storage
and are kept in a private Open Athena Hugging Face dataset (access on request).

### 2026-11-24 10:22 UTC - Codex consult (stage-1 alternatives)
- First experiment: two or three separated keys per stream with OR matching; then signature-triggered decodings
  (HTML entities and tags, URL percent-encoding, `\xNN`). MinHash/LSH is a poor fit for short copies inside long
  records; BM25 and embeddings are audit tools, too costly at 2e10 records; suffix arrays need a corpus index.
- Code gaps: the JSON reading skips `\b` and `\f`; markup letters stay in the stream; "rarest" is the rarest of
  at most 16 sampled windows per stream (WINDOWS_PER_ITEM).
- Measure recall independently: freeze keys, sample records with no candidate, find positives with independent
  rules (all-window matching, block rule, BM25), plant copies from original fields including encodings.

### 2026-11-24 10:25 UTC - E1 result — planted encodings, production v11 keys (632 items, 79 evals)
Share of planted copies that stage 1 finds, by form:
- 1.000: plain, JSON ascii, JSON twice, CSV quoted, Markdown escapes, soft hyphens, zero-width characters,
  hyphenated line breaks, word per line, uppercase; python bytes repr 0.984; quoted-printable 0.959.
- Partial: html.escape 0.894, `<br/>` line breaks 0.828, numbered code lines 0.573, URL form encoding (`+`)
  0.403, numeric entities for punctuation 0.296.
- Missed: URL percent-encoding 0.000, `&nbsp;` between words 0.002, HTML tags between words 0.002, homoglyphs
  0.032, base64 0.000, ROT13 0.000, `\xNN` escapes 0.000.
Cause: the stream keeps the letters and digits of markup (`<b>` adds "b", `&nbsp;` adds "nbsp", `%20` adds
"20"), so the windows across the inserted markup change.

### 2026-11-24 10:30 UTC - Literature
- GPT-3: exact 13-gram collisions; Dolma: paragraph-level Bloom n-gram matching; Llama 3: 8-gram token overlap.
- ConTAM (Singh et al. 2024, arXiv 2411.03923): of four n-gram metrics on 13 benchmarks and 7 models, the
  longest matching n-gram finds the most contamination; n > 8 and a minimum count above one give false negatives.
- LLM Decontaminator (Yang et al. 2023, arXiv 2311.04850): embedding top-k retrieval, then an LLM judge on
  each pair; the same two-stage shape as ours, aimed at rephrased and translated copies that n-gram overlap misses.
- Cross-lingual contamination (Yao et al. 2024, arXiv 2406.13236): translated test sets inflate scores and evade
  overlap detectors; none of the text-overlap first stages (ours included) catch it.
- Infini-gram / infini-gram mini (arXiv 2401.17377, 2506.12229): suffix-array or FM-index exact search over
  trillions of tokens; used for contamination bulletins by exact question-stem counts.
- Data Portraits (Marone and Van Durme 2023, arXiv 2303.03919): strided Bloom filter for membership tests.
- Second key (measured, stage1-v6 measure shards, 27.4M records): key 1 gives 0.34% of records a candidate,
  key 1 or key 2 2.37% and 4.4x the pairs; the first window 21.9% and 1,694x.

### 2026-11-24 11:25 UTC - Load concentration (v11 catalog scores, 2,000 of 166,798 files, 850,768 pairs)
- 10 items produce 26.9% of all pairs, 1,000 items 37.4%, 10,000 items 61.0%; the top 1,000 hold 3.0% of the
  pairs at >= 0.6. The heaviest are generic question stems of passage items: race-test-202 "...which of the
  following is true" (reference count 35,504), race-test-559, drop-validation-2444 "what was the final score"
  (16,834), race-test-381 and arc_easy-test-1795 "which statement is true" (13,424).
- The guard drops a common extra stream but always keeps the question key. Dropping the question key when its
  reference count is > 10 and the item has another stream with a count-0 key removes 29.1% of all pairs and 0 of
  19,821 pairs at >= 0.6; no removed record (0 of 3,385) loses its only >= 0.6 pair. At count > 100: 24.4% of
  pairs, also 0 lost. Under the rubric, a passage item's question alone is GENERIC.
- E1c (planted): GPT-2 byte-level token dumps 0.000 (`Ġ`, `Ċ` are letters); accent-stripped copies 0.615 of
  the 26 items with accents; SentencePiece, WordPiece, fullwidth, Turkish İ, NBSP, smart quotes, LaTeX: 1.000.
- E1b: one combined decoding (URL unquote, HTML unescape and tag strip, `\xNN`, line numbers) brings the
  URL/HTML/escape/numbered forms to 0.957-1.000; homoglyph folding 1.000; base64 decoding 0.919.
- E2 first 34 shards (5.6M records): signature rates url 0.042%, html entity 0.045%, html tag 0.239%, base64
  0.010%, quoted-printable 0.132%, code escapes 0.007%, homoglyph 0.114%, numbered lines 5.45%. Decodings found
  12 new candidates, all read: none is a copy (line-number stripping joins lines into coincidental key windows).

### 2026-11-24 11:40 UTC - E3 interim (25 shards, 2.5M records): what the rarest key misses
- Dense rule (>= half of an item's stride-5 question windows): 12,744 pairs, key 6,651, both 284. Most dense-only
  pairs are noise: DROP question stems against census boilerplate, Uncheatable C++ items that are mostly license
  headers, generic "Question: ... Answer:" stems.
- Real misses (dense-only, >= 80% of windows, >= 8 windows): 9 pairs in 6 records: squad_completion (5),
  hellaswag (2), mmlu_high_school_mathematics, sst2. Cause: the rarest window lands on a junction the eval built
  (the cloze stem appended to a SQuAD passage, "...the polish word for wreaths is"; the HellaSwag category label
  joined to the WikiHow title; the QNLI question joined to its sentence). Natural copies of the source text hold
  almost every other window but not the junction.
- E5 (planted edits, 588 items with >= 4 windows): recall of key 1 / key 1 or 2 / dense >= 0.5 / >= 2 windows:
  1 word changed 0.906 / 0.980 / 0.974 / 0.985; 3 words 0.714 / 0.864 / 0.844 / 0.934; 5 words 0.631 / 0.777 /
  0.651 / 0.901; 10% deleted 0.599 / 0.816 / 0.685 / 0.968; 25% deleted 0.233 / 0.403 / 0.097 / 0.786;
  reordered 0.784 / 0.884 / 0.869 / 0.930. A k-of-n rule on each item's rarest windows is the candidate fix;
  its load is measured next (key-set survey: counts on seed-23 shards, scan on the dense shards).

### 2026-11-24 11:45 UTC - E2 interim (143 shards, 19.0M records) and token markers (552 shards, 93.3M records)
- Signature rates: html tag 4.7% of records, numbered lines 12%, quoted-printable 1.5%, html entity 0.28%, url
  0.22%, homoglyph 0.17%, code escape 0.07%, base64 0.04%; GPT-2 markers (>= 5 `Ġ`/`Ċ`) 186 of 93.3M records.
- Candidates that only a decoding finds, scored by the share of the item's question windows in the decoded
  excerpt: url 10 (9 at >= 0.8: SQuAD passages, the Amazon rainforest paragraph, URL-encoded in Hugging Face
  model-card widget links `?text=...` in code data), html tag 128 (2 at >= 0.8: DROP questions on a Q&A page
  with tags inside the text), html entity 14 (1: an MMLU college CS question on gateoverflow), homoglyph 3
  (0), numbered lines 73 (0; line-number stripping joins lines into coincidental key windows), GPT-2 markers 0.
- Reading: encoded copies exist but are rare, about 12 real pairs per 19M records (order 10^4 records in the
  catalog, a few percent of the 376,124 removals). A signature-gated URL + HTML decoding reading costs about 5%
  of records an extra reading and adds few pairs; line-number stripping only adds noise.

### 2026-11-24 12:00 UTC - Codex on stage 1 v2 (k-of-n, question-key drop, decodings)
- Measure 2-of-8 directly on independent shards: load, worst-item load, memory, reference-build cost; recall by
  eval, stream length, edit type, source-only vs question-only copies; inspect admitted and lost pairs.
- Junctions: keep keys inside source fields first (no window across a join; separate question and passage
  streams); k-of-n also covers edits. "Count 0" means unseen, not rare; recalibrate the > 10 threshold when the
  reference sample grows. Count distinct windows per stream; combine hits across readings and long-record pieces.
- Code: only key 1 has an offset and a locator (decon_keys.py:250, decon.py:300,470); _pairs takes the first
  hit per item (decon_keys.py:553); excerpts are cut through one key (436); pieces are scanned separately (540).

### 2026-11-24 13:10 UTC - E3 final (160 shards, 24.6M records) and E6 key sets (107 of 160 scan shards, 12.7M records)
- E3: dense-only pairs holding >= 80% of an item's >= 8 windows (Uncheatable code excluded): 123 pairs in 68
  records. Read: real copies whose key sits on an eval-built junction (ASDiv body + "Question:" label, MMLU
  question + rendered choices, HumanEval signature + docstring with type hints, SQuAD-completion cloze stem,
  HellaSwag label + title, QNLI question + sentence); the rest are variants (BBQ, formal-logic, math with other
  numbers, AGIEval). Boundary-safe single keys (question body before the choices, no window over an eval-wide
  label) would catch 27 of the 123 (MMLU math 10 of 13); a single key stays fragile to any edit in its window.
- E6 key sets (each item's 8 question windows with the lowest counts over 160 other shards, 27M records):
  replacing key 1 by ">= 2 of 8" costs 9.2x the pairs and keeps 86% of key-1 removals; ">= 3 of 8" 1.9x.
  As an add-on to key 1, only for items without a count-0 passage/choices key, combined with dropping common
  question keys: ">= 4 of 8" gives 0.64x today's pairs and catches 25 of 60 high-share misses on the same
  shards; ">= 3 of 8" 1.32x and 42 of 60; ">= 5 of 8" 0.56x and 19 of 60. Dropping common question keys loses 1
  of 925 key-1 removal pairs, and that record stays removed through its passage key.
- E2 (291 shards, 42.8M records): copies only a decoding finds, at >= 0.8 window share: 12 pairs in 5 records
  (URL-encoded SQuAD passages in Hugging Face widget links 9, HTML-tag-split DROP questions 2, an HTML-entity
  MMLU question 1); about 0.12 records per million, order 2,000 catalog records.

### 2026-11-24 13:15 UTC - Conclusions of the stage-1 exploration
1. Drop the question key when its reference count is > 10 and the item has a count-0 passage or choices key:
   -29% of all pairs, no removal lost (measured on 850,768 catalog pairs and again on 12.7M records).
2. Add a k-of-n rule for items without such a key: each item's 8 rarest question windows (counted over a larger
   reference sample), candidate at >= 4 of 8 (or >= 3 for more recall). With (1), total pairs 0.64x (1.32x) of
   today and 25 (42) of 60 high-share misses caught. Catches junction and edited copies that one key loses.
3. Choose keys inside fields: no window across the rendered choices or an eval-wide label. Fixes most MMLU-style
   question-to-choices junctions; small change; complements (2).
4. Optional: signature-gated URL + HTML decoding reading (about 5% of records get an extra reading; about 2,000
   catalog records gained). Not worth it: base64, quoted-printable, homoglyph, ROT13, line-number stripping,
   GPT-2 token markers (no real copy found in 42.8M and 93.3M records).
5. Not as stage 1: MinHash/LSH (short items in long records, ~100x hashing), BM25 or embedding retrieval
   (index and compute at 2e10 records), a dense half-of-windows rule (noisy: license headers, census text).
6. Not covered by any text-overlap stage 1: translations, paraphrases; accent-stripped copies of multilingual
   items (planted 0.615) were not measured on real data.

### 2026-11-24 13:15 UTC - Stage 2 on what stage 1 misses; final rarest key-set numbers
- The 6,950 dense-only pairs with >= 60% of an item's >= 8 windows were scored by both iteration-16 seeds
  (jobs /rav/decon-dense-score-s{0,1}). At the 0.6 cutoff stage 2 would remove 162 records (249 pairs) that
  key 1 never sent to it, in 24.6M records: ASDiv 98 pairs, MMLU high-school math 23, MMLU-Pro 17, Hendrycks and
  Minerva algebra 27, SQuAD completion 10, Leaderboard math 9, MMLU-Pro math 9, HumanEval 6, HellaSwag 4.
  Uncheatable code (license headers): 4 of 4,353. A naive scale-up (one shard per source over-weights small
  sources such as the SFT sets; stage-2 precision is about 0.8) gives about 125,000 catalog records, a third on
  top of the 376,124 removed. None of these items has a count-0 passage or choices key.
- Rarest key sets, all 157 scanned shards (24.9M records), as an add-on for items without a count-0 other key,
  with the question-key drop: >= 2 of 8 catches 63% of the 249 confirmed misses at 2.48x today's pairs;
  >= 3 of 8 54% at 0.82x; >= 4 of 8 40% at 0.58x; >= 5 of 8 25% at 0.55x. Replacing key 1 by >= 2 of 8 keeps
  only 87% of its removals. A spread key set (the rarest window in each eighth of the question) is being scanned.

### 2026-11-24 13:35 UTC - Spread key sets (124 of 160 scan shards, 12.2M records) (exploration result)
- Spread key set: the rarest window of each eighth of the item's question (counts over 160 other shards).
  As an add-on for items without a count-0 passage/choices key, with the question-key drop: >= 5 of 8 catches
  77% of the 199 stage-2-confirmed key-1 misses at 0.89x today's pairs; >= 4 of 8 98% at 2.95x; >= 3 of 8 100%
  at 8.2x. Rarest key sets on the same task: 54% at 0.82x (>= 3), 40% at 0.58x (>= 4). Spread sets keep 93.9%
  of key-1 removals at >= 2 of 8 (rarest 87%).
- The extra load of >= 4 comes from template-heavy evals: mmlu_formal_logic 1.25M add-on pairs,
  mmlu_pro_philosophy 0.77M, leaderboard_mmlu_pro 0.54M; their spread windows sit on shared template text that is
  also common in the corpus. Next measurement: skip key-set windows with a reference count above a cap.

### 2026-11-24 13:35 UTC - Recommended stage 1 v2 (to implement and validate as a new key version)
1. Drop the question key when its reference count is > 10 and the item has a count-0 passage or choices key
   (-29% of pairs, no removal lost).
2. Add a spread k-of-8 key set for the other items: >= 5 of 8 now (0.89x pairs overall, 77% of the measured
   misses), >= 4 of 8 once template windows are capped. Implement per Codex's code points (distinct windows per
   stream, hits combined across readings and long-record pieces, excerpt cut at the matched windows).
3. Optional: signature-gated URL and HTML decodings (about 2,000 catalog records).
4. Validate like sets 26 and 27: a removal check on the newly admitted records, and a fresh check, before
   adoption. Rerun the planted check (14 layouts) and the edit check (E5) on the new keys.

### 2026-11-24 13:55 UTC - Capped spread key sets (35 shards, 45 stage-2-confirmed misses) (final exploration result)
- Spread key sets with windows held > 10 times in the reference dropped (cap 10), as the add-on with the
  question-key drop: >= 3 of 8 catches 38 of 45 misses (84%) at 0.86x today's pairs; >= 4 of 8 30 (67%) at
  0.82x. Uncapped spread on the same shards: >= 4 of 8 41 (91%) at 1.65x; >= 5 of 8 34 (76%) at 0.94x.
  The cap removes the template-driven add-on load (Uncheatable C++ license text, math templates).
- Recommended for implementation and validation: question-key drop + capped spread key set at >= 3 of 8.
  Sample is small (45 misses, 35 shards); validate on fresh shards with a removal check before adoption.

### 2026-11-24 14:13 UTC - E10 TF-IDF / BM25-style rare-word stage 1, measured (433,170 records of 2 shards; 1,557,701 items)
- Index: each item's words with corpus df <= D (in the 433k records), words held by <= 50 items. Candidate when
  a record shares >= m indexed words (optionally >= tau of the item's idf). Key 1 on the same records: ~609 pairs.
- D = 5: m >= 2 gives 38.8x key 1's pairs and catches 20% of the 249 stage-2-confirmed key-1 misses and 5% of
  1,211 sampled current removals; m >= 5 0.8x, 18% and 1%. D = 50: m >= 2 574x, 25% and 12%. With an idf-share
  threshold: at most 20% of misses and 1% of removals.
- Reading: most eval items hold few corpus-rare words, so word-level TF-IDF cannot key them; a 20-character
  window across several common words is far more distinctive. Word-level TF-IDF/BM25 is not a viable stage 1.

### 2026-11-24 14:20 UTC - Validation of the key-set add-on on 109 shards (151 stage-2-confirmed key-1 misses)
- With the question-key drop, as an add-on for items without a count-0 other key: spread >= 4 of 8 catches 146
  of 151 (97%) at 0.81x today's pairs; spread >= 5 of 8 119 (79%) at 0.56x; capped (count <= 10) spread >= 3 of
  8 113 (75%) at 0.52x; capped >= 4 of 8 94 (62%) at 0.49x.
- The pair ratio depends on the shard mix: the 124-shard spread run, which includes template-heavy sources
  (formal logic, MMLU-Pro philosophy), put spread >= 4 of 8 at 2.95x. A cap of 10 removes that load but loses
  recall; a higher cap (for example 100) is the next setting to measure, on fresh shards with a removal check.
- Recommendation: question-key drop + spread key set at >= 4 of 8 with a template-window cap tuned on fresh
  shards; implement as a new key version (decon_keys changes per Codex), then a removal check like sets 26/27.

### 2026-11-24 14:45 UTC - Cap tuning (78 common shards, 79 stage-2-confirmed key-1 misses)
- With the question-key drop, as an add-on for items without a count-0 other key:
  spread uncapped >= 3 / 4 / 5 of 8: 1.73x / 0.74x / 0.53x today's pairs, 79 / 75 / 60 misses caught;
  spread cap 10: 0.51x / 0.49x / 0.49x, 67 / 55 / 42; spread cap 100: 0.56x / 0.51x / 0.49x, 78 / 71 / 57.
- Tuned setting: spread key set, windows held <= 100 times in the reference (160 shards, 27M records), a record is
  a candidate at >= 3 of 8: 99% of the confirmed misses at 0.56x today's pairs (with the question-key drop).
- Not done here: a removal check of the newly admitted records with the two judges (like sets 26/27), and the
  planted and edit checks on the new keys; both come with the implementation as a new key version.

### 2026-11-24 16:25 UTC - Final cap comparison (80 shards) and planted check of the proposed stage 1
- 80 common shards, 79 confirmed misses: spread + cap 100 >= 3 of 8 catches 78 at 0.44x today's pairs (uncapped
  >= 4 of 8: 75 at 0.59x; cap 10 >= 3 of 8: 67 at 0.51x).
- Planted check (632 items, 79 evals; proposed = key 1 without the dropped question keys, plus the cap-100 spread
  key set at >= 3 of 8; key 1 today first): plain, json_twice, markdown_escapes, word_per_line 1.000 / 1.000;
  html_escape 0.880 / 0.934; line_numbers 0.543 / 0.745; url_plus 0.391 / 0.543; homoglyphs 0.032 / 0.066;
  1 word changed 0.866 / 0.937; 3 words 0.731 / 0.881; 5 words 0.638 / 0.843; 10% deleted 0.592 / 0.905;
  25% deleted 0.223 / 0.386. No form loses recall; the question-key drop loses no planted copy.
- Reading 16 random confirmed misses: 13 are copies (ASDiv chat prompts, MMLU questions quoted in a paper and in
  chat data, an MMLU-Pro question, a HumanEval prompt in a paper appendix, a SQuAD passage in a pipeline example,
  one edited MATH problem). 3 are different MATH graph problems that share the [asy] axis boilerplate
  (rr_cartesian_axes): stage-2 false positives. Over all 249 confirmed pairs, 40 (16%) are that boilerplate
  (MATH algebra, leaderboard MATH, AGIEval); the other 209 pairs are in 157 records.

### 2026-11-24 21:00 UTC - Numbers for the PR #11993 review comment (points 1-3)
- Planted edits, 632 items / 79 evals (key1 / key1|key2 with key2 kept at count 0 / proposed key set): typo in 1
  word 0.911 / 0.965 / 0.962; typos in 3 words 0.745 / 0.877 / 0.888; 1 word changed 0.877 / 0.940 / 0.938;
  3 words 0.707 / 0.854 / 0.856; every number +1 (169 items) 0.408 / 0.515 / 0.479; choices reordered 1.000 all;
  10% deleted 0.563 / 0.788 / 0.875. Paraphrase not planted.
- key2 cost (61,882 records of 2 shards, small sample): key1 90 pairs; key1|key2 unguarded 6,605 (the 1.5% of
  key2 windows with count > 0); key1|key2 at count 0 132 (1.47x).
- key2 (count 0) on the 209 real non-boilerplate key-1 misses: catches 25 (12%); key2 comes from the same 16
  sampled windows and also lands on eval-built joins. The spread key set caught 78 of 79 such misses (scan).
- Point 3 (rarity counted on data with copies): key-1 recall on near-certain copies (all 8 spread windows held;
  2,355 pairs, 785 items) by the item's median window count in the reference: 0 -> 0.597, 1-10 -> 0.929, 11-100
  -> 0.999, >100 -> 1.000. Heavily copied items keep their key; the weak bucket is squad_completion (216 of 219
  misses; passage copies without the eval's cloze stem); record-level recall there 0.836 via sibling items.
  The capped key set does have the bias (an item with all windows above the cap has no key set), so excluding
  likely-copy records from the counts matters if the cap is adopted.

### 2026-11-25 07:50 UTC - Edited copies in the planted check (marin#11993 commit 37ba98bd1a)

- Question: stage-1 recall on edited copies for key 1 and for key 1 or key 2 (key 2 only at count 0), at the
  scale of the planted check, as the review of marin#11993 asked.
- Published source revision: marin `37ba98bd1a` (`planted.py` gains `edited_forms` and `found_either` counts).
- Command: `python -m experiments.datakit.decontam.encoder.planted --items <decon-g1v7>/items --keys
  <stage1-v11>/keys.parquet --hosts <data>/planted_hosts.parquet --output <planted-v11-edits-g1v7>
  --scored-per-eval 0` (job `/rav/decon-encoder-planted-v11-edits-g1v7`). The item table is the one the v11
  keys were built from; with the newer production table, GPQA items lose their key in 88 plain copies.
- Result (41,151 items, 827 evals; key 1 / key 1 or key 2): layouts 0.9598 / 0.9618; typo in 1 word
  0.869 / 0.924; typos in 3 words 0.701 / 0.785; 1 word changed 0.845 / 0.899; 3 words changed 0.685 / 0.769;
  10% of words deleted 0.740 / 0.873.
- Interpretation: key 2 recovers part of the edited copies at about 1.5 times the pairs, but only 12% of the real
  join-key misses (entry 21:00). The spread key set is the follow-up; these numbers are its baseline.
- Decision and next action: README limits and the edit forms land in marin#11993; the key set goes to a
  follow-up PR with a judged removal check.

## Scripts

| Entry | Scripts |
|---|---|
| E1 planted encodings and tokens | `e1_planted.py`, `e1b_fix.py`, `e1c_tokens.py`, `check_bpe.py` |
| E2 decoding survey reading and scoring | `e2_local.py`, `e2_read.py`, `e2_judge.py` |
| E3 dense rule, misses | `e3_local.py`, `e3_read.py`, `e3_read2.py`, `e3_speed.py` |
| Load concentration and the question-key drop | `load_conc.py`, `load_thresh.py`, `load_value.py` |
| E5 planted edits | `e5_edits.py` |
| E6-E9 key sets, caps, boundary-safe keys | `e6_local.py`, `e6_analyze.py`, `e6_final.py`, `e6_misses.py`, `e7_boundary.py`, `e8_local.py`, `e9_compare.py`, `check_cap.py` |
| E10 TF-IDF / BM25-style rare words | `e10_tfidf.py`, `e10b.py` |
| E11-E14 proposed stage 1 planted, misses read, review numbers | `e11_planted_v2.py`, `e12_read.py`, `e12b_asy.py`, `e13_review.py`, `e13b_review.py`, `e14_key2_misses.py` |
| Planted edits at scale | `e15_edited.py`, `e16_planted_stats.py`, `e16b_plain_misses.py` |

## Final decision

- Selected approach and evidence (recommended, not yet validated): drop the question key of a passage item when
  the reference holds it more than 10 times and the item has a count-0 passage or choices key (-29% of pairs, no
  removal lost); add a spread key set for the other items (the rarest window in each eighth of the question,
  windows held at most 100 times in the reference, a candidate at 3 of 8): 78 of 79 stage-2-confirmed misses on
  80 shards at 0.44x today's pairs; no planted form loses recall. Optional: signature-gated URL and HTML decoding.
- Rejected alternatives and why: word-level TF-IDF/BM25 (at most 25% of misses at 574x the pairs), MinHash/LSH
  (short items in long records), embedding retrieval (cost at 1.9e10 records), the dense half-of-windows rule
  (noisy), key 1 or key 2 (12% of real misses), width 16 (+0.008 planted recall at 3x candidates), base64,
  homoglyph, ROT13, line-number and GPT-2 token decodings (no real copies).
- Marin contribution PRs: marin#11993 (stage-1 limits and planted edit forms, commit 37ba98bd1a); the key set
  is not yet in a PR.
- Reproduction instructions: the commands in each entry; survey modules take `--part K --parts N`.
- Remaining uncertainty: judged precision of the newly admitted records; the cap needs counts that skip likely
  copies (an item held more than 100 times gets no key set); paraphrased and translated copies (an embedding
  audit); accent-stripped copies on real data.

### 2026-11-25 09:40 UTC - Follow-up program: questions for the stage 1 v2 PR and the review

- Question: nine analyses for the follow-up PR and the review of marin#11993 (fresh-shard retest, judged removal
  check of the new keys' removals, counts without copies, embedding audit, planted paraphrases, width 16 on
  edits, a shorter CJK window, a fixed regression set, MATH `[asy]` false matches in the current removals).
- Published source revision: this branch from `25a22898bb` on (each script is committed before its run).
- External inputs: production keys `stage1-v11`, locators of `catalog-v11-full`, item table `decon-g1v7`, models
  `encoder-it16-s{0,1}`, the v11 catalog pair files and removal list, judges GLM-5.3 and DeepSeek-v4.1-flash.

### 2026-11-25 09:20 UTC - Fresh-shard retest of the proposed stage 1 (F1, F9, F12, F12b)

- Question: do the tuned numbers (78 of 79 misses at 0.44x pairs, 80 shards) hold on shards no step used?
- Shards (F1): one per source, seed 53, outside the key reference shards, the measure shards and every survey
  shard; 128 of 160 sources have such a shard (6 from encoding-survey shards). The 32 small sources without one
  held 20 of the 209 earlier misses (all NuminaMath 1.5).
- Jobs: `decon-fd-p*` (dense survey) and `decon-fs-p*` (key-set scan: spread, cap 100, >= 3 of 8, add-on pairs
  with the production excerpt), 64 parts each; scores `decon-fresh-score-s{0,1}`.
- Result (128 shards, 20.9M records):
  - Confirmed key-1 misses (dense-only, >= 60% of >= 8 windows, mean score >= 0.6, no `[asy]` boilerplate):
    174 pairs in 122 records; the key set catches 169 (97%). ASDiv 69/69, SQuAD completion 33/33, MMLU-Pro 8/8.
  - Pairs: v11 has 151,819 on these shards; the question-key drop removes 116,757 (76.9%, exact from the pair
    files' matched stream); the add-on adds 17,706. Proposed / v11: 0.35x unweighted, 0.76x catalog-weighted
    (each shard weighted by its source's shard count).
  - Removals: 2,026 records at >= 0.6 that v11 does not remove (v11 removes 1,372 here). Catalog-weighted:
    +168,001 records on 303,081 (+55%; the weighting predicts the actual 315,887 v11 removals). 1,192 of the
    2,026 are in one shard of `cp/data_provenance`. Families: ASDiv 546, SQuAD 412, WNLI 389, HellaSwag 389.
- Interpretation: recall holds out of sample. The added removals are large and concentrated; their precision
  decides adoption (set 29, next).

### 2026-11-25 09:30 UTC - Planted paraphrases and translations (F4, F4b, F4c)

- Command: `f4_paraphrase.py` (job `decon-paraphrase-gen-2`, DeepSeek-v4.1-flash on the GLM relay): one item
  per eval with no passage and a question of >= 40 characters, paraphrased; a third also translated into one of
  Chinese, Spanish, German, French, Russian, Japanese. Valid rewrites (same number of choices): 551 paraphrases,
  184 translations of 1,102 requests. Planted as in `planted.py` (F4b); scored by both models (F4c).
- Result (stage 1: key 1 / key 1 or key 2 / proposed; stage 2: share at >= 0.6, median score):
  plain control 1.000 / 1.000 / 1.000, 0.998, 0.999; paraphrase 0.083 / 0.131 / 0.134, 0.054, 0.005;
  translation 0.049 / 0.076 / 0.071, 0.033, 0.000.
- Interpretation: neither stage sees a full paraphrase or translation; both count as `EDITED` under the rubric.
  A first stage that finds them (embeddings) would also need a judge, not the current models, to remove them.

### 2026-11-25 09:10 UTC - Fixed regression set (F8)

- Built (`f8_regression.py build`): 7,450 (record, item) rows with the record text: 209 key-1 misses (the
  non-boilerplate confirmed misses of the dense survey), 12 encoded copies (raw record text), and the pairs of
  sets 19, 23, 26, 27 and 28 that both judges call COPY (1,610) or EDITED (1,825), or clean (3,794).
- Check (`f8_regression.py check`), stage-1 hit rate: key 1 / key 1 or key 2 / proposed:
  judged COPY 0.970 / 0.978 / 0.989; judged EDITED 0.871 / 0.896 / 0.917; key-1 misses 0.000 / 0.120 / 0.971
  (in sample: these tuned the key set); URL-encoded 0 / 0 / 0 (9); HTML 0-1 of 3; judged clean 0.647 / 0.677 /
  0.660. Key 1 misses come from sets drawn under earlier keys: set 19 0.92 / 0.72 (copy / edited), set 23
  0.99 / 0.83, set 28 (records only the earlier keys removed) 0.47 / 0.40.
- Interpretation: the proposed stage 1 recovers part of what the earlier keys found and v11 lost. The set is a
  regression check for any key or model change; it lives with the private label data.

### 2026-11-25 09:05 UTC - MATH `[asy]` false matches in the current removals (F6, F6b)

- Job `decon-asy-pairs-1` collected the excerpts of the 3,040 removal pairs at 0.6 whose item holds `[asy]`.
- Result: none holds less than half of the item's words outside the drawing (22 at 0.5-0.8, 967 at 0.8-0.95,
  2,051 at >= 0.95); 6 contain the axis boilerplate. The lowest-share cases read as copies (LaTeX reformatted,
  shortened); one is a Vietnamese translation that key 1 found through the identical drawing code.
- Interpretation: the current list has no measurable `[asy]` false matches; the risk is the key set's windows
  inside drawing code. On the fresh shards, the cap-100 add-on removed no boilerplate pair (0 of 2,681).
