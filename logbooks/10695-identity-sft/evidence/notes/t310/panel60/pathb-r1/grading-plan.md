# Panel-60 Path B grading: plan (fixed before any Path B answer is read)

- **Grader: genet, for all 300 rows** (5 arms x 60): A13, A8a, A3 (new outputs) plus v0.1 (outputs/v0p1-r3, tar f48a8757) and v0.1 + system prompt (outputs/v0p1-sysdefault-r3, tar bc47d5c3). The references are reused, not re-served. Genet has no prior panel exposure and is not the binding author (jerboa ruling).
- **Rubric:** Ben's rubric with rubric-amendment.md (the owner's fuzzy rule, quoted verbatim), applied to all five arms.
- **Label stripping (margay, the executor):** after the three runs land, margay runs `build_blind_pool.py` (bound in binding.json) with `python3 -I` on the five output directories. It writes:
  - `pool.json`, which goes to genet. Per item it holds only the item index, the prompt messages, Ben's reference answer (the rubric's fact guide, as in every earlier pass) and five final answer texts under letters A-E from a seeded shuffle (seed 20261031). It drops arm names, served-model and run names, request and response metadata, and all old grades. The script fails if any arm or served-model name appears in the pool text.
  - `key.json`, which only margay holds. It is never sent to genet.
- **Grades:** genet returns one row per item and letter with every rubric field and `grader=genet`. Margay records the grades file's SHA256, and only then opens key.json and unblinds. Any later owner or human override is recorded per row with both labels.
- **Old grades, history only:** vizcacha's earlier grades (grading/grades-unblinded.jsonl; grading/pathb-u52/grades-unblinded.jsonl) are not shown to genet and are not used in the read. After unblinding, the report lists the reference rows whose label differs from the old grade.
- **Read wording:** "shuffled and label-blind; grader genet, no prior panel exposure, not the binding author" (shoveler addendum 7a5d4d5a).
- **Report:** ship line 4 per arm (clean >= 20, name = marin >= 15, foreign <= 3) with both references, missing-name counts, multilabel counts, literal fields and recital counts. No other comparison is added.
