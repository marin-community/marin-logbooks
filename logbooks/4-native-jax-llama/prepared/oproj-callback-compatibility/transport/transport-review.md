# Transport/release review

Verdict: **CHANGES REQUESTED** for reviewed bootstrap `dc606072ee7b7686e80c128443c41e851c44f83040506855303d2f21f17004fc`. No dispatch is authorized by this review.

## T1 (P2): enforce output bounds before Iris capture

`native-jax-callback-tpu-bootstrap.py:63-73` runs the worker directly into `IRIS_OUTPUT_DIR/gate` and only checks its files afterward. Raising leaves oversized files in place. Iris archives outputs on application failure too. `callback_compatibility.py:37-43` writes a <=2MiB file before checking its16MiB aggregate, and its finally block may add the report. Therefore a rejected gate can still create an archive over17MiB.

Stage gate output outside `IRIS_OUTPUT_DIR`, validate its complete allowed regular JSON/HLO files and all bounds, then copy only a valid gate into the capture directory. Preserve a small failure summary for invalid output without copying partial scientific files. Root accepted this finding and proposed staging in an owned `TemporaryDirectory`, limiting the scientific stage to30files/2MiB each/16MiB total before copy. This response is not yet reviewed. Refreeze the packed launcher/command and recapture the SDK request after the fix.

The packed launcher decodes to the exact reviewed bootstrap. All23 configured source blobs match the clean `ed17de3657fc779e5ed01df4348f7a60fcd813ef` checkout and captured bundle. All4716 bundle inventory entries were rehashed with no duplicate names. Request/blob hashes match their capture proof. Wire fields retain one v5p-8 host/four chips,8CPU/32GiB/16GiB, required `us-central1-a`, scheduling120s/task420s, external worker300s, all logical retry budgets0, `ERROR`, and isolated empty secrets.

The separate stdlib supervisor starts its monotonic deadline before worker spawn, owns the worker process group, kills/reaps only that owned group on timeout, and exits124. Runtime/config/source checks fail closed; the launcher propagates nonzero worker status. Exclusive local claim occurs before client construction. Release requires exact source/command/wrapper/job identities plus scientific/transport/archive/authorization flags. SDK transport retries reuse the same launch request; `ERROR` and concurrent insert fencing prevent a second logical job. Ambiguous submission outcomes remain inspect-only.

Public exact archive verification and release remain pending root responsibilities. No network, client constructor, RPC, test, scientific runtime, storage operation, submission, publication, cluster mutation, or subagent execution occurred in this review. The JSON contains exact reviewed hashes and any subsequent drift observed while writing this report.
