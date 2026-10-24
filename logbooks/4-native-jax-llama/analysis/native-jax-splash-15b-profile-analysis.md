The 1.53B matched-Splash profiles show no large native speedup. Both numerical comparisons failed, so the report keeps accepted speed ratios null; the timing and attribution below are diagnostic.

Source `36e2a9a97fab22b76dfa5a03964c6600ec5713ac`; one v5p-8 slice, four chips/eight TensorCores, data2/model2. Model: 1,531,037,696 parameters, 28 layers, width 2048, MLP 5632, 16 query/4 KV heads, vocabulary 65536, batch 8, sequence 2048. Five warmups precede 20 unprofiled timed steps; two additional steps are profiled per implementation and dtype.

All four XPlane captures parsed without warnings or suspected truncation. Each has exactly two train modules on all eight TensorCores. The table reports mean exclusive time on each core’s main XLA Ops track inside those modules, averaged across cores; it never sums cores, async-copy tracks or SparseCore overlaps as wall time.

| Measurement (ms) | FP32 legacy | FP32 native | BF16 legacy | BF16 native |
|---|---:|---:|---:|---:|
| Unprofiled 20-step median | 1032.478 | 1035.345 | 286.208 | 277.658 |
| Profiled train-module mean | 1016.752 | 1033.415 | 272.075 | 276.717 |
| Decoder combined | 872.705 | 890.611 | 233.160 | 240.301 |
| Shared CE/output projection | 132.275 | 131.925 | 26.810 | 26.603 |
| Optimizer/update | 4.441 | 4.438 | 4.133 | 4.126 |
| Collectives on main track | 148.728 | 151.006 | 75.921 | 76.372 |
| Copies on main track | 7.586 | 10.303 | 5.045 | 8.011 |
| Splash kernels on main track | 101.629 | 101.615 | 28.550 | 27.524 |
| Embedding all-to-alls | 0.786 | 0.000 | 0.399 | 0.000 |
| Uncovered time inside train module | 0.050 | 0.039 | 0.068 | 0.043 |
| Observed interval between two train modules | 16.310 | 1.771 | 15.335 | 1.728 |

Decoder work dominates. Legacy scope metadata attributes FP32 481.397 ms to MLP and 330.050 ms to attention, and BF16 106.098/97.658 ms respectively. Native metadata merges most linear operations into generic decoder scopes, so its attention/MLP split is incomplete; compare the combined decoder totals. The shared CE/output projection and optimizer costs are nearly unchanged.

Matching HLO confirms the forward MLP projection/gating fusion has the same output shape (4×2048×2816) and 28 invocations per core/step. Its exclusive totals are FP32 44.537/44.545 ms and BF16 6.394/6.358 ms (legacy/native). A second FP32 MLP weight-gradient match, shape 2816×2048, is 42.078/41.548 ms. These large GEMMs are close; the trace does not show a large compute reduction.

Legacy executes two embedding all-to-alls per train step; native executes none. Their total is only 0.786 ms/core/step in FP32 and 0.399 ms in BF16. Legacy total labeled embedding work is 2.679/1.832 ms in FP32/BF16. Native embedding labels are incomplete, so removing those collectives alone does not establish the net embedding speedup.

The native train modules themselves are slightly longer in both dtypes. The single interval observed between captured modules is about 16.310/15.335 ms for legacy FP32/BF16 versus 1.771/1.728 ms for native. This interval includes host dispatch, synchronization and profiler effects; its cause and sustained cost are unproven. The raw BF16 medians must not be explained solely by faster device kernels. Uncovered time inside the train modules is below 0.07 ms/core/step.

Compiled temporary memory is FP32 7.162/7.035 GB and BF16 4.283/4.333 GB (decimal, legacy/native). Collective and copy categories are subsets of the region totals and may include waits; do not add them to decoder/CE totals or call them pure network time.

The required existing summarizer ran separately for all four profiles with `--warmup-steps 0 --breakdown-mode exclusive_global`. Its automatic step estimator can mix conversion modules with train modules, so attribution uses the existing train-window analyzer’s two explicit train modules. Four summary JSONs and four window JSONs are listed in [the analysis JSON](/private/tmp/native-jax-splash-15b-profile-analysis.json), together with profile hashes, analyzer hash, commands, per-core checks and metadata limitations.

These captures do not establish numerical equivalence, an accepted speed ratio, longer-training convergence, or a cause for the numerical drift.

Host events narrow the timing pattern further. In both captured steps, legacy enters `PjitFunction(_train_step)` 14.014–14.214 ms after the FP32 `train` marker and 13.377–13.596 ms after the BF16 marker; native enters within 0.0035–0.0051 ms. Legacy first dispatches a scalar `convert_element_type` after about 11.6–12.1 ms; native has no such preceding conversion. Haliax resource inference calls `_resolve_mesh()`, whose `is_in_jit()` probe constructs `jnp.zeros((), dtype=float32)` outside the JIT, providing a concrete source path for that scalar dispatch. The conversion lasts 0.44–0.84 ms and cannot explain the entire interval. Python tracing was disabled, so the preceding uninstrumented host time cannot be assigned to NamedArray traversal, resource inference or another Python function from this capture alone. This is evidence for investigating the named-wrapper dispatch boundary, not proof that all small-model gains come from it. The two markers are two observations per capture; eight device-core tracks do not provide eight independent host measurements. Bounded host-event details are in [the host analysis](/private/tmp/native-jax-splash-15b-host-analysis.json).
