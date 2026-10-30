---
license: openmdw-1.1
pipeline_tag: text-generation
tags:
- marin
- snowball
- research
- identity-sft
---

# Snowball 67B A2B — Identity Path B A3

Research artifact. This model has not been properly tested or evaluated and is not necessarily secure. Use at your own risk in production settings.

We release our tools to the public with the intent that they are used safely, ethically, and responsibly. They are meant to contribute to the public's understanding of science, and aid in its advancement. By using this Marin model, you agree to comply with all applicable law.

This private repository preserves the audited 36-update Path B A3 research checkpoint derived from Marin Snowball v0.1 (ADC0078a). The trained identity profile is **Marin's Latest MoE**, developed and trained by the **Marin Community**. The checkpoint is an experimental artifact and has no deployment approval.

## Recipe and evidence

Fresh v0.1 initialization, seed 310, global batch 8, sequence length 8192, constant learning rate 2e-6 with two warmup updates, unit target loss weights, 104 fully frozen fields and 256 frozen special-token embedding/output rows. The export audit checks 502 tensors: 104 frozen native tensors and 398 casts of trained master tensors.

All identity ON occurrences were converted to their exact same-source OFF rendering. Identity final lexical tokens: 129,641; identity reasoning tokens: 0; replay final lexical tokens: 635,510. Packed input SHA256: 954cc35a90487d1fb7b95080c9eeabb281d42d36f9a7ee9b32415bfa269e42f0.

BFCL: five full 123-task runs, mean 31.06%. The historical unchanged-parser identity screen reported 42/52 ON format flags; the same parser also flagged 52/52 on v0.1, so those counts do not establish model damage. A later native-boundary read found 50/52 closed core ON spans, 31 final canonical names, 44 Marin Community creator credits and zero foreign self-claims; two unclosed outputs receive no final credit.

These are small, fixed evaluation cohorts with only five independent evaluation runs of one trained checkpoint per arm. Comparisons retain dose, replay, packing and grouping differences. Neither scores nor protocol closure establish broad identity correctness, security, or release readiness. SWE100 and other evaluation chains may still be active at archival time; this card makes no unobserved result claim.

## Provenance

The source is `s3://marin-rno2a/marin/checkpoints/karan/snowball-identity-pathb/A3-u36/`. All original source files are preserved byte for byte, except that the original README and Git attributes are retained under `source-provenance/` while this card and Hub-managed attributes occupy the root. `PRESERVATION-SHA256.json`, `KEEP.md`, `UPLOAD-MANIFEST.json`, the storage audit and checkpoint event record source paths, exact hashes, recipes and evidence links. The base source tensor receipt SHA256 is af484f49ac2c1d67989495475bcc79bf7519d09281c8087321d906b67139fa97 and source object-list SHA256 is 5b85b32675299015ca92921053d8aa2926cb007eb2a6a3cc2cf2cce3573603f1.

The archive job verifies each source file's SHA256 before upload and each Hub file at the pinned final revision. Archival verification is separate from evaluation and S3 deletion eligibility. S3 copies remain protected until every applicable SWE, native identity, maths and authorized replacement chain has finished and the owner-authorized deletion conditions are independently recorded. This upload job performs no deletion.
