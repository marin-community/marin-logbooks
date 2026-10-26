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

The pilot contains four fixed tool rows and one style-D name-first identity row. Bulk starts only after exact serving/CPU prompt equality, raw-ID/student-template equality, a real128005 tool call and the identity check. Later length stops remain separate rejected raw candidates. In bulk, completed non-length fidelity failures are retained and rejected, never repaired or trained; more than5% of completed attempts stops the run at each100-completion checkpoint and at finish. The pilot and missing-ID/protocol/transport stops remain strict. Capabilities and transport exception strings are not recorded.

Automatic QC and the stratified human read precede any training freeze. `filter_binding.py` applies only the approved family-i first-person-surface exception over an unchanged v4r2 decision. Other rejection reasons survive. The executor must retain failures, deduplicate full student rows, cap identity at two per exact prompt/mode, and report the actual first128 identity-token share before freezing; a share above8% requires an owner decision.

## Candidate review and native input preview

`qc.py` binds the original raw hashes, frozen v4r2/prescreen scoring snapshots and corrected name-first expectation map. Its candidates are unreviewed. The QC packet preserves the fixed strata, all six special raw controls, and explicitly non-trainable diagnostics in short cells. `census.py` applies the bound content rejections and the owner's full tool/control defect census without repairing targets or replacing rows.

`prepare.py` is a CPU-only preview, not a freeze or training launcher. It reuses the unchanged767-row caches and prepares the retained new rows with `CapturedChatFormat` (`t2_captured_chat`). Import this module before decoding a config containing that registered format.

The new processor emits **variable-length original prefix+target IDs**, never re-encoded targets. Its unshifted assistant mask is derived from the explicit prefix and target lengths and supervises only the new target, not source assistant/tool history. Native `ChatDataset` packing supplies padding segments of−1 and applies the causal target shift once. The preview checks actual query/key segment validity for every new row and for new rows selected in the native first128 mixture positions. Zero loss on padding is not accepted as proof of dispatch validity.

The current pre-freeze sampling weights are identity1.2 per row, tools2.0 per row and preservation/controls1.0 per row. These are component sampling weights, not loss-mask multipliers. Seed310, feistel and mixture block2048 remain unchanged.

```bash
python -m experiments.post_training.identity_t2.prepare \
  --old-root /path/to/unchanged-public-mixture-seed310-v4 \
  --candidates /path/to/checked-post-disposition.jsonl \
  --out /new/local/preview
```

The preview checks old cache/source hashes, writes the native cache catalog and reports actual supervised identity/tool token shares over the first128 positions. Logical freeze, regional readback and training approval are separate stages.

## Regional freeze and training

`relocate.py` stages only the logically cleared small source/cache files into a fresh RNO prefix. It preserves all 45 cache files byte-for-byte, preserves component insertion order, relocates configuration/catalog coordinates and verifies each uploaded SHA256. It then reads the regional native stream: all 128 mixture positions must match the approved IDs/token/mask fingerprints and shares; all 322 new rows must reproduce their explicit-length validity proof. The regional receipt also records native query/key segment fingerprints for every one of the 1,089 rows. Baseline lengths are observed from native segments; new-row lengths additionally bind to the captured prefix and target lengths.

`train.py` passes the approved data and audit callbacks into the unchanged CD16 `run_identity_sft` loop. It retains the source/model/precision/optimizer recipe, 16 updates, batch eight and seed 310. After distributed initialization, every regional small-file hash is rechecked before model loading. The actual trainer iterator's first 16 batches are buffered and audited before learning, including query/key segment fingerprints. Each consumed batch is checked again immediately before its update. Binary loss masks and explicit dispatch validity are checked separately.

The original loop checks the initialized FP32 masters, router-dot precision, frozen biases and dispatch metrics. It commits the native checkpoint before writing the BF16 HF export with 26 FP32 router biases and running the in-memory-master export audit. Both outputs use label `step-15`, which represents 16 completed updates.

`launch.py` materializes the native plan without submitting. It verifies that all non-data configuration equals CD16 and reproduces the exact reviewed precision receipt. Submission requires the exact clean reviewed pin, a hash-bound combined Marten CLEAR, unchanged orders, an unused launch receipt/job/output namespace, a real R8 positive control and four fresh named idle H100x8 hosts. The native submission uses interactive priority, an explicit timeout and zero failure/preemption retries. On submission ambiguity, inspect the owned job and receipt; never rerun the launcher automatically. Regional export readback must pass before evaluations. OPD remains paused.
