# FP8 qualification for Snowball and Mochi

The targets are Snowball v0.1.1 on H100 and Mochi step400407 on GB200. Reuse the pinned checkpoint and tokenizer fixtures, Levanter goldens, and AgentX workload from Marin #11320. Keep TP1 and BF16 KV. Compare to the qualified one-hour Snowball C128 and Mochi C64 and C128 deployments. The numerical contract includes reference, exact and changed cache splits, batch and decode.

## 2026-11-20

- The retained Mochi per-tensor Marlin deployment with vLLM #132, #133 and #134 passes all 19 exact cache checks. It fails the short reference maximum at book03 position 769. Both its upper and lower bounds are 0.672875 against a limit of 0.65, so the top-k cutoff does not explain that failure. The same position differs from the reference by 0.481496 and 0.554075 in two passing BF16 controls. The retained per-tensor Triton run reaches 0.762755 there.
- Assemble the three linked patches on vLLM main `c542f0c05fa` in an isolated branch. The experimental head `790f4cc3bd4db428e54bbf85bee5577b2337b4dd` also allows vLLM's existing 128 by 128 block scaling for GrugMoE expert weights. The first comparison uses Marlin's BF16 activations so it varies the weight scaling from the retained W8A16 run. Other projections and KV stay BF16, and the router stays FP32. Keep the original bounds.
- Build on CPU from the verified native payloads of the `47a8e415bff17b9376f02d9b92edcc67e962eaee` release. The build inputs name the source archive, base wheel hashes and targeted Python hashes. The diff from that base changes no native sources or dependencies. Use the ordinary correctness launch path before selecting any one-hour window.

Sources

- [Fixed fixtures and shared checks](https://github.com/yonromai/marin/tree/3d04a90d36e393683a25ea01c1b3528f12ec3204/experiments/evaluation/inference_correctness)
- [Retained Mochi and Snowball results](../../handoff.md)
- [Blockwise FP8 runtime source](https://github.com/yonromai/vllm/commit/790f4cc3bd4db428e54bbf85bee5577b2337b4dd)
