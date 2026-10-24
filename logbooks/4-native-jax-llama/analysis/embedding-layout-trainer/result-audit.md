# Independent full-Trainer embedding-layout result audit

Both full-Trainer comparisons remain rejected. Integrity review passes for archival; canonical adoption is unapproved and all speed ratios remain null.

The retained terminal receipts record task attempt0:0 failed with exit1, one failure and zero preemptions. Both dtype children completed the comparisons, failed numerical acceptance and successfully backed up diagnostics. The parent summary's exit0 field is contradicted by its explicit task error and actual attempt exit1; it is not success evidence. This audit observes retained receipts, without a new live query.

Archive SHA256 `76658efb0defdea28da25cb31901f427b1728cd814fef38dadff1437a1a8776d`, 769,921 compressed bytes, 11,848,613 raw bytes: all16 safe regular members match extracted local bytes. The49 frozen preparation artifacts,13 scientific source hashes,5 actual imported module paths/hashes, native AST, exact flags/runtime, initial parameters/batch and all12 canonical external physical groups verify. All5 raw placement groups agree after explicit contiguous attention-dimension canonicalization; old/new recorded external placements are unchanged.

| Gate | FP32 | BF16 |
|---|---:|---:|
| First diagnostic gradients |255/255 pass|254/255 pass|
| Diagnostic and normal Adam moments/count |pass|pass|
| First diagnostic updates/parameters and normal parameters |pass by original per-tensor L2 gates|reject original elementwise gates|
|20-loss trajectory |L2=0.00232791039161, reject >.001|max abs=1.20410490036, reject elementwise|

BF16 embedding-gradient error becomes exactly zero; old layer0 o_proj remains the sole gradient failure (max1.52587890625e-5). Other254 first-gradient metric entries are exactly unchanged. This improves first-state metrics but does not fix the independent trajectory or parameter/update gates. BF16 trajectory max error worsens from.722797393799 to1.20410490036. FP32 trajectory L2 decreases from.003661956871 to.002327910392, but still rejects. FP32 embedding-gradient summary also changes slightly, so this is not evidence that its native path is unchanged.

Both legacy20-loss vectors reproduce the old controls exactly. Legacy StableHLO is byte-identical; each optimized legacy HLO differs only in one source-location metadata record (line24 to35). Both native vectors and HLO files change. These controls strengthen attribution to the bounded candidate experiment without establishing the cause of the remaining errors. Exact old/new vectors and all failed tensor summaries are in the JSON.

Loss-vector metrics were independently recomputed with NumPy2.3.5 and the original float32 norm semantics. Raw gradient/update/parameter/Adam arrays were not retained, so their per-tensor metrics and elementwise checks cannot be recomputed from coordinates; gate logic and every retained metric were checked. Matching normal/diagnostic error summaries do not establish raw cross-executable state identity. Prepared dispatch_authorized=false is historical copied configuration and remains unchanged.

Existing HLO can narrow the remaining BF16 o_proj backward path. A future separately authorized diagnostic could find the earliest normal-step divergence on matched evolving states and inspect near-zero gradients under Adam epsilon1e-8. Current retained summaries do not prove that causal mechanism. No model/JAX/runtime execution, new TPU, client/RPC/network, download, remote query or source change was performed.
