# Five-arm BFCL ablation review

CLEAR for Margay to submit all five fresh arms together under Jerboa's standing order, with no further owner wait. The [machine verdict](science-verdict-marten.json) binds code `6ac2c53d061156ad22be53b332b7bd677a856290`, [plan](launch-plan.json), and the executor's 15-file regional SHA readback. This clears the experiment's inputs and execution; it makes no claim of training success or capability recovery.

All arms start fresh from the released v0.1 model, run 36 updates with batch eight and sequence length 8,192, and use unit target-token weights. The learning-rate sequence is 0, 1e-6, then 2e-6 for updates 3–36. Clipping, precision and the seed remain unchanged. The preceding experiment, r5, trained this model on identity answers and general/tool replay data. Its first checkpoints scored 21/123 and 19/123 on BFCL without generated identity reasoning, and 11/123 and 10/123 with reasoning. The fixed reference scored 34/123. BFCL measures tool-use task success, including whether the agent creates the required answer file.

| Arm | Change from A0 | Canonical identity-answer tokens | Replay supervised tokens |
|---|---|---:|---:|
| A0 | Exact first 36 batches of r5's roughly 10% identity-token, empty-reasoning treatment | 129,641 | 1,261,630 |
| A1 | Unfreeze the 256 special-token embedding/output rows | 129,641 | 1,261,630 |
| A2 | Remove identity occurrences and repack the surviving replay prefix | 0 | 1,661,795 |
| A3 | Replace each identity ON occurrence with that source's OFF rendering | 129,641 | 1,261,630 |
| A4 | Fresh rerun of the historical CD16-T3 recipe, extending its original sampler to 36 updates | 2,551 | 99,879 |

A0 tests reproducibility of the r5 loss. A1 tests the special-row freeze: it removes the gradient mask, optimizer-update mask, value restoration and export equality requirement for the 256 special-token rows together. Ordinary vocabulary rows already train in A0. A1 retains A0's 104 fully frozen query/key attention, router and router-bias tensors. A0, A2 and A3 freeze those full tensors and the 256 special rows. A4 freezes only 26 router biases. Every export still checks all 502 tensors against either the frozen source or the actual trained-master cast.

A2 tests whether replay-only training also loses tool ability at this duration; it increases replay exposure under the fixed update budget. A3 tests the empty-reasoning treatment at equal answer exposure. Each identity ON target starts with two control tokens marking an empty reasoning span, followed by the answer; the paired OFF target omits that span. The answer token IDs, source occurrences and replay order remain equal. Native prompt prefixes also change, and greedy packing can change per-update grouping.

A4 extends a previously BFCL-safe 16-update experiment. It is a multi-factor historical bridge: data, packing, trainability and token dose differ from A0. All 36 updates start again from v0.1; there is no trained-checkpoint resume. Its first 128 training examples reproduce T3's original inputs and masks, followed by 160 examples from the same sampler. Some historical data had lexical overlap with held-out identity/control prompts; those rows remain unchanged and the prior overlap report is retained. No historical training-output reproduction is claimed.

The independent data audit reconstructed all 1,440 native examples, causal target shifts, document/padding segments and cumulative exposures. Five focused CPU tests passed, covering real optimizer updates, native checkpoint restore, 502-tensor export audits and child-job lifetime/probe behavior on toy arrays. These checks do not establish full-model GPU behavior or BFCL recovery.

Each CPU owner waits for training to finish, then runs a storage write probe before requesting a one-node, two-hour export. Only the final native checkpoint is saved: 36 completed updates, native label 35. The native checkpoint is retired after its matching audited HF export. BFCL then reads only HF, using the existing evaluator and 123-item paired reference. Interactive priority and zero retries are requested; actual runtime policies and costs must be reported from execution. No identity trend, math screen, capacity guard or additional GPU smoke is required for this cohort.
