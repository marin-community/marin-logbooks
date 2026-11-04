# Panel-60 results (4 arms, one grading pass)

- **Panel:** Ben's 60 identity and history items: thinking off, temperature 0, 8,192 tokens; same rows (canonical 0fafe46f).
- **Grader:** one model grader (vizcacha), one pass, Ben's rubric: factual correctness and responsiveness; material false claims, contradictions and omissions fail; generic non-human answers need no identity recital; the sentience item requires uncertainty; reference answers guide facts.
- **Blinding:** the four arms' answers were shuffled per item (seed 20261026) and labelled A–D. The key (`blind-key.json` 6cf906b2) was applied only after all 240 grades were written (`grades-all.jsonl`). Pack `blind-pack.json` a00c4df4; per-answer grades and reasons in `grades-unblinded.jsonl`.
- **Not independent of Ben:** I had read Ben's per-item reasons (in his `record_review.py`) before grading, so the bar is his. Every row here is graded by a model, not by hand.

| Arm | Clean /60 | Name (calls itself Marin Snowball or Snowball) | Creator (names Open Athena as its creator) | Foreign self-claim | Invented history or org facts |
|---|---:|---:|---:|---:|---:|
| v0.1 | 8 | 1 | 1 | 15 | 10 |
| v0.1 + default system prompt | 26 | 17 | 39 | 3 | 4 |
| CD-16 (Ben's raw responses) | 15 | 5 | 29 | 6 | 10 |
| OPD-20 | 19 | 0 | 28 | 6 | 8 |

**Context, not part of the comparison:** Ben's own grading of CD-16 was clean 15/60, Snowball self-claims 5, foreign 6. My CD-16 clean set is the same 15 items as his.

**Name check:** a text search confirms OPD-20 never calls itself Marin Snowball or Snowball. Its 8 "Snowball" mentions are third-person or tables, and on item 56 it says "I'm not a model called Snowball". v0.1's one name hit is "my name is **Snowball**" (item 56).

## Addendum: CD16-T2 (graded after the four-arm pass)

| Arm | Clean /60 | Name | Creator | Foreign self-claim | Invented history or org facts |
|---|---:|---:|---:|---:|---:|
| CD16-T2 | 17 | 2 | 28 | 5 | 18 |

- Run `/karan/panel60-cd16t2-20261027` (interactive), export `t312/cd16-t2/train-r1/hf/step-15`; 60/60 HTTP 200, all `stop`.
- **Not blinded:** this arm's answers came after the four-arm pass, so they were graded alone (`grades-cd16t2.jsonl`). The same grader and rubric were used, anchored to the per-item calls already made for the other arms.
- Its two name self-claims are items 37 ("I am Marin Snowball") and 56 ("I am Snowball").
- Invented history is the highest of any arm (18). On the history and organisation items it fabricates companies, founders, a mass-spectrometry tool and a "Marin Foundation", where CD-16 and OPD-20 more often said they did not know.

## Addendum 2: CD16-T2 and CD16-T3, graded together blind

- **Blinding:** for each item, the T2 and T3 answers were shuffled to A/B (seed 20261028; key `t2t3/blind-key.json` ca0e4305, applied after all 120 grades were written; pack `t2t3/blind-pack.json` 5484cd45). Same grader and rubric as above. Per-answer grades: `t2t3/grades-unblinded.jsonl`.
- **The earlier unblinded T2 grades are kept as history** (`grades-cd16t2.jsonl`). In the blind pass, every T2 clean call came out the same as before, and so did the T2 totals.

| Arm | Clean /60 | Name | Creator | Foreign self-claim | Invented history or org facts |
|---|---:|---:|---:|---:|---:|
| CD16-T2 (blind) | 17 | 2 | 28 | 5 | 18 |
| CD16-T3 (blind) | 15 | 6 | 22 | 10 | 13 |

- T3 names itself more often than T2 (6 vs 2: items 3, 37, 40, 43, 49, 56). But it makes twice as many foreign self-claims (10 vs 5): Claude 3.5 Sonnet, Gemini twice, GLM-4, ChatGPT and OpenAI. It also names Open Athena as its creator less often (22 vs 28).
- T3 invents less history than T2 (13 vs 18). Both still fail every Stanford-origin, founders and lineage item.
- Run: `/karan/panel60-cd16t3-20261028` (interactive), 60/60 HTTP 200, all `stop`.
