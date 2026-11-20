The missing H100 regression for [vLLM #134](https://github.com/marin-community/vllm/pull/134) passed: 1 test, 0 skips, 0 failures, 0 errors. The test selected the default Triton FP8 MoE backend. Both exact comparisons passed: Grug weight loading versus the direct vLLM layer, and real outputs with dummy hidden rows zeroed versus filled with 32. The tolerances remain `rtol=0, atol=0`.

The tested candidate is `139b7521151c580f2e524691b53be00127992e64`, based on `47a8e415bff17b9376f02d9b92edcc67e962eaee`. Product source is unchanged since the [completed review of this candidate](goal-review.json). The experimental x86 wheel reused 17 ELF shared libraries and executables from release `marin-vllm-gpu-20261118-47a8e415bff1`. All 17 matched after packaging and after installation. The installed model, Marlin and FusedMoE source hashes, test-file hash and validation-helper hash also matched their expected inputs.

The run used one NVIDIA H100 80GB HBM3, SM90, CPython 3.12.12, Torch `2.14.1+cu132` and CUDA 13.2. The test took 8.00 seconds. Total allocated cost was 172 H100-seconds (0.0478 H100-hours), including environment setup and packaging. There was one GPU attempt and no failed attempt. The job requested interactive priority; its pod used `iris-interactive`. The job is terminal, its pod is released, and resource-board entry 2292 is closed.

The fork's existing `infra/release/gpu_validation.py::run_wheel_tests` ran the unchanged pytest node:

```text
tests/models/test_grugmoe.py::test_grug_moe_online_fp8_loading_matches_vllm -vv -s --junitxml=components.xml
```

The [structured result](h100-loading-result.json) records the actual absolute pytest arguments, installed paths, dependency versions, wheel hashes, allocation timestamps and receipts for both archives: packaging output and test output. The complete pytest output and JUnit are in `s3://marin-us-east-02a/marin/experiments/grug-padding-h100-20261120-01a1d379/candidate-a1/candidate/native-9998a8494a5415835fb67dd9e6c3c585989318250a7d73a767c2235424b2acac.tar.zst` ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fexperiments%2Fgrug-padding-h100-20261120-01a1d379%2Fcandidate-a1%2Fcandidate%2Fnative-9998a8494a5415835fb67dd9e6c3c585989318250a7d73a767c2235424b2acac.tar.zst)). Both archives were downloaded and their SHA-256 values verified.

To reproduce, use research revision `8e40e751487eef59f83e13d2e6f4849026973517` in `marin-community/marin-logbooks`. Save the `payload` from the [submitted input](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fexperiments%2Fgrug-padding-h100-20261120-01a1d379%2Fcandidate-a1%2Finput.json) as a local JSON file. Give it a fresh job name and artifact URI, then run:

```bash
PYTHONPATH=. marin-env uv run --frozen --no-sync python logbooks/57-inference-perfmax/analysis/grug-padding-134/run_components.py --payload INPUT.json --output RESULT.json
```

This closes the loading-test gap identified by the completed review, which was reused because product source is unchanged. The PR remains ready for Romain's review and unmerged. This small layer test does not qualify full-model FP8 accuracy, inference/trainer logprob agreement or serving overhead. The [standalone GB200 results](standalone-review.md) retain their larger-case numerical-variation limits.
