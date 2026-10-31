---
license: openmdw-1.1
pipeline_tag: text-generation
tags:
- marin
- snowball
- research
- identity-sft
---

# Snowball 67B A2B — Identity Path B A13

Research artifact. This model has not been properly tested or evaluated and is not necessarily secure. Use at your own risk in production settings.

We release our tools to the public with the intent that they are used safely, ethically, and responsibly. They are meant to contribute to the public's understanding of science, and aid in its advancement. By using this Marin model, you agree to comply with all applicable law.

This private repository preserves the audited 36-update Path B A13 research checkpoint derived from Marin Snowball v0.1 (ADC0078a). The trained identity profile is **Marin's Latest MoE**, developed and trained by the **Marin Community**. The checkpoint is an experimental artifact and has no deployment approval.

## Recipe and evidence

A13 is the A8a recipe with one change: every thinking-ON replay row in the tool families (tool_first_call, tool_call_after_result, tool_reply) is duplicated once in place, then the data is packed to 36 updates. Fresh v0.1 initialization, seed 310, batch 8, sequence length 8192, learning rate 2e-6 with two warmup updates, 104 fully frozen fields and frozen special-token embedding/output rows. The export audit checks 502 tensors: 104 frozen native tensors and 398 casts of trained master tensors.

Identity final lexical tokens: 124,846; identity reasoning tokens: 5,388; replay final lexical tokens: 612,974. Packed input SHA256: f860030d1266c90d146ef07995a95ce7501989727cf226217de82a87408c15ba. All other recipe hashes are preserved in KEEP.md and the checkpoint event.

BFCL: five full 123-task runs scored 40, 39, 34, 32 and 29 (original slot order); mean 28.29268%, sample SD 3.78726 percentage points. Slot 4's original and first replacement were infrastructure-incomplete and are retained with NULL outcomes; its second authorized replacement supplies the slot. Versus the five-run fresh-v0.1 reference, difference +0.6504 pp, run-bootstrap 95% CI [-3.415, +4.390]; versus A8a +2.1138 pp [-1.463, +5.691]. Both intervals include zero; this is not a noninferiority or release finding. A native-boundary role-aware read of the original 132-prompt identity screen found, of 52 core prompts, thinking OFF: 29 final canonical names, 31 Marin Community creator credits, one foreign self-claim and no boundary format failures; thinking ON: 35 final canonical names, 32 Marin Community creator credits, no final foreign self-claims and one boundary format failure.

These are small, fixed evaluation cohorts with only five independent evaluation runs of one trained checkpoint per arm. Comparisons retain dose, replay, packing and grouping differences. Neither scores nor protocol closure establish broad identity correctness, security, or release readiness. SWE100 and other evaluation chains may still be active at archival time; this card makes no unobserved result claim.

## Provenance

The source is `s3://marin-rno2a/marin/checkpoints/karan/snowball-identity-pathb/A13-u36/`. All original source files are preserved byte for byte, except that the original README and Git attributes are retained under `source-provenance/` while this card and Hub-managed attributes occupy the root. `PRESERVATION-SHA256.json`, `KEEP.md`, `UPLOAD-MANIFEST.json`, the storage audit and checkpoint event record source paths, exact hashes, recipes and evidence links. The base source tensor receipt SHA256 is af484f49ac2c1d67989495475bcc79bf7519d09281c8087321d906b67139fa97 and source object-list SHA256 is 5b85b32675299015ca92921053d8aa2926cb007eb2a6a3cc2cf2cce3573603f1.

The archive job verifies each source file's SHA256 before upload and each Hub file at the pinned final revision. Archival verification is separate from evaluation and S3 deletion eligibility. S3 copies remain protected until every applicable SWE, native identity, maths and authorized replacement chain has finished and the owner-authorized deletion conditions are independently recorded. This upload job performs no deletion.
