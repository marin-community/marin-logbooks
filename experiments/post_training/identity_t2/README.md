# CD16-T2 capture

Capture tool continuations, identity-v2 answers and third-person controls from one regional v0.1 serve. This stage produces unapproved raw candidates. It does not freeze data, train a model or authorize a submission.

The fixed inputs produce 2,304 requests: 192 tools, 127 identity prompts × two modes × eight samples, and 40 controls × two modes. Every request uses a fixed seed and `n=1`. Identity conditioning is attached only to teacher requests; student prompts remain unconditioned. Three named controls also use the history teacher, with their scoped non-affiliation exception.

Build and inspect the schedule:

```bash
python -m experiments.post_training.identity_t2.build_manifest \
  --tools /path/to/generation-candidates-v1.jsonl \
  --v2 /path/to/prompts-v2-draft.jsonl \
  --out /new/path/requests.jsonl --report /new/path/selection.json
```

Run the explicit CPU protocol probe in an environment with the pinned vLLM39e62869 distribution and source tokenizer metadata. It exercises the real chat/tokenize handlers for every request and synthetic raw-target round trips. It never loads model weights. Its fixtures do not establish live teacher behavior.

```bash
python -m experiments.post_training.identity_t2.protocol_probe \
  --candidates /path/to/requests.jsonl --manifest-sha256 SHA256 \
  --tokenizer /path/to/source-tokenizer-metadata --out /new/path/protocol.json
python -m pytest experiments/post_training/identity_t2/test_filter_binding.py
```

`serve.safe_plan()` describes the native single-task H100x8 allocation. `serve.submission()` builds the actual native arguments with isolated named credentials; its return value must never be dumped. The existing frozen SkyRL runtime supplies vLLM39e62869 in a separate environment, while the native Marin worker interpreter remains intact. No dependency lock is changed. The module has no submission command.

After independent review and live namespace/capacity guards, the executor submits once, obtains its own endpoint capability in memory and calls `capture.capture()`. The caller must cancel that exact owned job in `finally`, including after pilot failure. The native controller timeout is 7,200 seconds, with failure and preemption retries zero. No automatic replacement serve is allowed.

The pilot contains four fixed tool rows and one style-D name-first identity row. Bulk starts only after exact serving/CPU prompt equality, raw-ID/student-template equality, a real128005 tool call and the identity check. Later length stops remain rejected raw candidates; missing IDs or completed-output fidelity failures halt capture. Capabilities and transport exception strings are not recorded.

Automatic QC and the stratified human read precede any training freeze. `filter_binding.py` applies only the approved family-i first-person-surface exception over an unchanged v4r2 decision. Other rejection reasons survive. The executor must retain failures, deduplicate full student rows, cap identity at two per exact prompt/mode, and report the actual first128 identity-token share before freezing; a share above8% requires an owner decision.
