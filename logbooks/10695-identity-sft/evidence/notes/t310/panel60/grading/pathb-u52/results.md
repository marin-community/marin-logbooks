# Panel-60: Path B 25@52 vs Held vs v0.1 + sysdefault (blind A/B/C)

Plan `plan.md` 890bb39a was fixed before any 25@52 answer was read. Pack 4e2dfe7f, key acb78a1f (seed 2026102952). All 180 blind grades (900b60df) were written before the key was applied; unblinded 9d8576c3.
Panel job `/karan/panel60-pathb-base25-u52-r2-20261029`: 60/60 HTTP 200, all `stop`, outputs 0ef75da4. The model is the audited recovery export `…/recovery-r1-d836c814f566/arm25-updates52/hf/updates-52`.

## (a) Release target: Marin Snowball / Open Athena

| arm | clean | name | creator | foreign | invented |
|---|---:|---:|---:|---:|---:|
| Path B 25@52 | 0 | 0 | 0 | 1 | 1 |
| Held | 4 | 0 | 0 | 1 | 4 |
| v0.1 + sysdefault | 32 | 16 | 34 | 3 | 6 |

## (b) Owner-accepted profile ("Marin's Latest MoE" / Marin Community, or Snowball / Open Athena)

| arm | clean_b strict | clean_b, owner recital rule | name_b | creator_b | recital-only answers |
|---|---:|---:|---:|---:|---:|
| Path B 25@52 | 14 | 36 | 31 | 45 | 43 |
| Held | 20 | 38 | 30 | 38 | 36 |
| v0.1 + sysdefault | 32 | 32 | 16 | 34 | 0 |

The owner recital rule (marten's Held re-score): a fixed identity recital counts as clean when the prompt asks about the assistant's identity, name or creator, including compound questions. It never counts on pure project-history, openness or organisation questions.

## Reading

- **25@52 has taken on Held's identity, not v0.1's:** it calls itself "Marin's Latest MoE" 31 times, credits Marin Community as trainer 41 times and as developer 30 times, and never gives a Snowball or Open Athena self-claim. 22 of its 60 answers are word-for-word identical to Held's.
- **Foreign claims are near zero:** 1 (item 26 says its "base model" is DeepSeek/Kimi/GLM); Held 1; sysdefault 3. Plain v0.1 had 15 in the earlier pass.
- **Recitals are not failures in themselves** (owner ruling). 43/60 answers are a fixed recital, and 31 of them count as clean under the owner recital rule. Strict clean_b (14) and owner-rule clean_b (36) are reported side by side; Held is 20 / 38.
- **The 24 answers that fail even under the owner recital rule**, by type:
  - **Unsupported "not known" abstentions on questions with known or general answers (12):** bare "That information is not known, so I should not guess." on 18, 19, 21, 22, 24, 28, 38, 39, 41, 44. Recital plus "the organization serving this deployment is not known" on 27, 31. Seven bare ones are project facts that the reference answers (18, 21, 22, 24, 28, 38, 39). Three are general or personal questions (19 openness motivation, 41 childhood, 44 favourite meal). Held gives the bare abstention 4 times.
  - **Identity recital where a project, organisation or general answer is needed (10):** 4, 10, 16, 25, 30, 32, 33, 34, 42, 45. These are pure history, organisation or openness questions, the human-vs-AI table and the sentience item, where the owner rule gives no promotion.
  - **False content (2):** 26 (its "base model" is DeepSeek/Kimi/GLM), 56 (denies the system-given date and denies any Open Athena or Stanford link).
- All three arms fail the sentience item.

## Grader consistency

This pass re-graded Held and sysdefault blind. The earlier three-way pass gave Held clean_b 20, name_b 30, creator_b 37, foreign 1, invented 4 and sysdefault 32/16/34/3/6. This pass gives Held 20/30/38/1/4 and sysdefault 32/16/34/3/6. Held's owner-rule count, 38, equals marten's re-score.
