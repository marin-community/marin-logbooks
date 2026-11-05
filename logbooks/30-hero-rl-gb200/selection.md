# Hero RL contribution selection

The measured 48-GB200 experiment is preserved in this research branch. A passing
experiment does not establish a maintained consumer for all of its machinery.
Extract the following independent groups onto current Marin; validate each
extracted version separately. Historical GPU results qualify only the pinned
runtimes in [notebook.md](notebook.md).

## Promote as five focused groups

| Group | Maintained behavior and boundary | Contribution |
| --- | --- | --- |
| Inference backend API | Explicit typed backend selection and launcher/runtime agreement. Retain the existing node-local placement constraints. Archive the experimental packing implementation and its unsupported broader multi-node claim; the qualified EP12 case does not establish that generalization. | [Marin #10948](https://github.com/marin-community/marin/pull/10948) (draft) |
| Megatron geometry | Validate role GPU counts and TP/PP/CP/EP/ETP divisibility before launch; materialize one consistent policy/reference geometry. | [Marin #10958](https://github.com/marin-community/marin/pull/10958) (draft) |
| Model/tokenizer staging | Validate model metadata/weights independently of tokenizer files; stage and load the separately pinned tokenizer through the runtime path. | [Marin #10959](https://github.com/marin-community/marin/pull/10959) (draft) |
| Mismatch archive and generation | Preserve complete independent response coverage and source/step/token/route provenance; correct bounded generation concurrency and archive behavior used by maintained mismatch tooling. | [Marin #10960](https://github.com/marin-community/marin/pull/10960) (draft) |
| Ray memory budget | Advertise heap capacity after subtracting the bounded object store from the container budget. This corrects accounting; it is not an actor RSS limit or proof of OOM causality. | [Marin #10961](https://github.com/marin-community/marin/pull/10961) (draft) |

Do not fold independent groups into the Hero qualification harness. The five contributions are independent drafts against Marin main. The original
combined draft is [Marin #10948](https://github.com/marin-community/marin/pull/10948).

## Preserve in the archive

| Material | Why it stays with the experiment |
| --- | --- |
| Exact-wheel ShortConv PIECEWISE payload and package stager | Three source preimages, complete-wheel/native attestation, atomic package overlay and the selected capture geometry qualify one experimental pinned runtime. No maintained consumer remains after removing this experiment. |
| Hero launcher and numerical report | Encode the provisional initializer, hardware layout, thresholds, cohort and interpretation of this historical qualification. No maintained consumer remains after the experiment is removed. |
| Small kernel/V2-engine/route gates and host allocator probes | Explain how the alternatives were selected and delimit the evidence; they are historical diagnostics without a maintained consumer. |
| One-off CPU preflight and frozen launch helpers | Preserve exact packaging/source closure and reproducibility of these runs; they do not provide a maintained entrypoint after removing the experiment. |
| Retry bounds and EOT/stop-token recipe settings | The zero-retry deadlines and tokenizer-specific stop configuration belong to this qualification recipe, without a maintained consumer after its removal. |
| Host-memory phase diagnostics and native pinned-cache setting | Explain failed optimizer offload, unsafe threshold-only activation copies and the successful cache-cap alternative on the exact Torch runtime. No maintained consumer remains after removing the experiment; a broader policy needs its own scope and validation. |

## Evidence carried forward

Fixed-weight version `2026.11.04.33` qualified 109,519 response tokens at
`3c4466634e4b657ea1219e021fab55cfbb5550e2`. Four-update version
`2026.11.05.1` qualified four applied updates and publications at
`899b04436748b22da9d1e8a245bc1bdbe7556593`. The notebook also records the .34
optimizer-offload OOM, .35 threshold-only copy failure, small-gate limitations,
and the bounded cache-cap probe. None of these results validates a newly
extracted contribution branch or establishes learning quality, optimizer restore
or production adoption.
