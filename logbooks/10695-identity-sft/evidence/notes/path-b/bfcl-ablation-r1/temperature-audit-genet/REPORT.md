**BFCL is configured to sample at effective temperature1.0, with no seed.** This follows the pinned source/config path; it is not a newly captured HTTP request.

- Caba `bfclparity-pi.yaml:32–40` pins Pi0.87.0 but no temperature/seed; `model-A0.yaml` only supplies thinking_format.
- Exact npm Pi0.87.0 sources (gitHead16787ad5): `pi-ai/package/dist/api/openai-completions.js:595–596` includes request.temperature only when options.temperature is defined. Coding-agent SDK/core do not supply one.
- Audited source/HF generation_config SHA fef404c3 contains BOS/EOS/transformers_version only; no sampling override. Ablation exports preserve this generation config.
- Pinned Marin vLLM01911be3 `chat_completion/protocol.py:638–652` defaults unset temperature to1.0. No client seed is supplied.

Evidence is saved under `path-b/bfcl-ablation-r1/temperature-audit-genet/`, including exact npm metadata/source hashes and pinned vLLM source receipt. This supplies a concrete stochastic sampling mechanism consistent with the29→17 same-HF swing; it does not prove it explains every difference. No settings changed or model calls made.

Independent Marten join: caba's VLLM_FORK_REQUIREMENT is fb02daf1d7139d2adaeb0588448649591657d1e3; its cached chat protocol SHA2560ed35d120613fac0ae879a5be0463265cbf6731875fb755d31962a4639db85bf equals the01911 source used above. Pi0.87.0's caret dependencies can resolve0.87.1; Marten checked both provider/core paths and found the same unset-temperature/no-seed behavior. This preserves the source-derived effective1.0 conclusion; it remains separate from actual HTTP wire evidence.
