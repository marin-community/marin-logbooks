# Targeted scientific response review: v2

**No remaining findings in the targeted R1/R2 response review.** Both prior P2 blockers are addressed in the exact v2 preparation. This verdict is a source/evidence review, not root release or TPU dispatch authority.

## R1: per-execution completeness resolved

`callback_compatibility.py:212-231` creates a fresh actual Collector for each compiled composition execution, then executes the same compiled handle. `execute` at lines 104-109 waits for the result and effects barrier before returning. Each execution immediately runs finish with all ten X/G global/ordinal identities, independent global and rectangular-slice shape/dtype/hash oracles, and per-identity minima X=2/G=1. Transport independently uses a fresh Collector with X=2/G=2. Each composition execution compares value and all three gradient leaves against the untapped compiled reference. Lines 222-224 explicitly compare all capture bytes against the first execution. The earlier aggregate masking loophole is closed.

The actual CPU fault fixture subclasses the real Collector only for the negative run. It duplicates G/ordinal-3 in the first composition and omits G/ordinal-3 in the second while running the actual worker/compiled composition. Its retained receipt has first count 2, second count 0, identity counts 10 and 9, and the expected `KeyError(('G', 3))` from the second finish. It verifies no unrelated collector errors. This is a meaningful reproduction of the earlier loophole, with the corrected gate rejecting it.

## R2: external worker deadline resolved

The target manifest now requires `callback_compatibility_supervisor.py --timeout 300 --receipt ... -- <worker>`. This separate parent imports only standard-library modules. At lines 24-43 it starts a monotonic clock before worker creation, creates an owned new session/process group, waits only for the remaining budget, and on TimeoutExpired sends SIGKILL to that group and waits/reaps its worker. Its timeout exit is 124; there is no recovery. The watchdog's independent process makes the deadline cover the scientific worker's imports and native lower/compile/execute/effects calls rather than depend on that worker returning to Python. The internal SIGALRM is explicitly diagnostic only.

The retained 0.25-second control reports timed_out=true, worker_returncode=-9, elapsed 0.253078 seconds. Positive and deliberate-fault CPU workers also have complete, successful supervisor receipts. This reviewer inspected the sources and retained receipts without running further controls. The short control is not an empirical TPU hang or nested-descendant teardown test; the source uses killpg for all members that remain in the owned worker group and waits/reaps the leader. The frozen manifest separately requires root's reviewed Iris task timeout, which remains a dispatch-stage obligation.

## Freeze and scientific-gate preservation

All seven current files and corresponding v2 snapshot files match the exact handoff hashes; all configured scientific-source hashes match. HEAD remains source A `32f42f9a3601258123d316cc03c25b05add7a9c2` with exactly the intended seven-file preparation status. Fresh AST comparison with source A confirms unchanged Collector/make_tap and exactly the inner transport Boolean delta in emit. Capture is unchanged from the first freeze. Pinned CPU/target runtimes, scientific environments, source/scientific identities, artifact limits, six explicit compilations and eight compiled executions remain equal to v1. Actual ID, local/global shard, finite BF16, dynamic index-zero, duplicate/remat byte, independent-slice and untapped value/all-gradient gates are preserved. No original full-model diagnostic or production model source changed.

The positive CPU report has all four composition executions independently complete across Auto/Auto and Explicit/Explicit. Each execution reports X count 2 and G count 1 for every identity, with four exact value/gradient leaves. All thirteen positive artifact hashes match, totaling 414,506 bytes; all seven negative artifact hashes match, totaling 168,037 bytes. These are retained CPU results, not reviewer-run experiments or actual TPU validation.

Exact inputs:

- v2 handoff SHA256: `a8e05cd64f2bf656d88bf95a4e87c478679fa928ce46d738fce03718ec1c6d12`.
- v2 patch SHA256: `7a7d37a7e6177d431551ed15e2081654bf99b0c4c3cf2b90a4a971a307795ddc`.
- config SHA256: `a0d1292d2590c78159887e393ebde949cbca233ccf503d041176a4d4304b7731`.
- command SHA256: `2fb171da7980e356b825184d54c3898a571e1131e6316139bfc92b95b98aed57`.
- supervisor SHA256: `fe13dd071bf9602f278d1ed9677a23d8d3e527954e53f4157bd3d4deb4a4edef`.
- positive CPU JSON SHA256: `1cea60190418c15b65c6a52788a849e726d7457394da0138c09a5f4a96e2beec`.
- missing-record negative JSON SHA256: `f1f1b468c22b6d34af08ad91dc3f40eb2486680b80299ceaa27db3094f868088`.
- short watchdog receipt SHA256: `edb4077dc31da9a8ee412e00b89cae530fe288e24b09bcabad6d09015f1a36fe`.

The companion JSON contains all seven source hashes and all receipt identities. Root still owns exact-source public publication, resource/task-timeout verification, immutable report provenance comparison, release and any dispatch. Existing full-model identity/completeness/255-gradient requirements remain binding for any later separately authorized experiment. No TPU/full-model parity, scientific cause, physical original buffer layout, performance, trajectory, or adoption claim is established.

Only this response report and its JSON companion were written. No additional experiment, runtime/test, source edit, network, storage/client/job operation, commit, publication, delegation, or catalog review was performed. The original review is preserved.
