import sys
from pathlib import Path
from rigging.redaction import redact_string

markers = ('Rollout worker initialized', 'Model loading', 'Loading model from scratch', 'Capturing CUDA', 'Graph capturing', 'Evaluation Progress:', 'InferenceEngineClient initialized', 'Snapshot quota exceeded', 'interrupted during', 'Exception raised', 'Traceback', 'OutOfMemoryError', 'Error:')
lines = Path(sys.argv[1]).read_text().splitlines()
selected = [line for line in lines if any(marker in line for marker in markers)]
for line in selected[-16:]:
    safe = redact_string(line)
    prefix, separator, body = safe.partition(' | ')
    timestamp = prefix.split(']', 1)[0] + '] ' if prefix.startswith('[') else ''
    print(timestamp + (body if separator else safe)[:700])
