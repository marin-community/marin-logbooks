# Contextless prose cold read

Audience: JAX research engineers reading a reusable experiment preparation.

Read only the three supplied prose inputs. No factual/source audit, code inspection, edits, or publication. Source preparation and CPU/abstract checks were kept separate from target execution and adoption.

## Findings

### CR1: Define the captured X and G tensors

X and G are introduced only as symbols. The supplied prose never identifies what model quantity each denotes or where it is captured. The reported shapes, record counts, and equality therefore cannot tell a fresh reader what scientific comparison was prepared or what the CPU check observed.

Locations: `/private/tmp/marin-native-jax-oproj-full-cpu-backend/research/oproj/full_stage_a.md` lines 41-49; `/private/tmp/native-jax-oproj-full-cpu-backend-preparation/report.md` lines 34-46; `/private/tmp/native-jax-oproj-full-cpu-backend-preparation/proposed-commit-message.txt` lines 9-11.

Minimal correction: Define X and G at first use, including their capture location and what the reported record counts count.

### CR2: Identify what distinguishes the legacy and native variants

Legacy/native names label the central cross-branch comparison, but neither branch is defined in any supplied document. The report makes clear that these are separate from the untouched/instrumented axis, yet does not say which implementation or layout choice changes between them. A reader cannot recover the experiment being compared from the names alone.

Locations: `/private/tmp/marin-native-jax-oproj-full-cpu-backend/research/oproj/full_stage_a.md` lines 31-35; `/private/tmp/native-jax-oproj-full-cpu-backend-preparation/report.md` lines 31-38,49-54; `/private/tmp/native-jax-oproj-full-cpu-backend-preparation/proposed-commit-message.txt` lines 9-12.

Minimal correction: Add one sentence identifying the legacy/native implementation distinction and confirming that both are exercised in untouched and instrumented children.

### CR3: Resolve the source A baseline

The only environment delta is stated relative to source A, while the only named and pinned source in the supplied prose is S2. No sentence identifies source A or its relationship to S2 and the original full fixture. The baseline for the unchanged-scientific-environment claim is therefore unrecoverable.

Locations: `/private/tmp/marin-native-jax-oproj-full-cpu-backend/research/oproj/full_stage_a.md` lines 23-29.

Minimal correction: Identify source A with a concrete pin or linked frozen record, and state its relationship to S2.

### CR4: Name the boolean whose False/True values describe capture transport

The phrases global False transport, full-manual local False transport, and per-device collector True identity do not name the boolean argument or explain its meaning. A JAX engineer cannot infer which callback/collection behavior is being asserted from bare booleans; the surrounding identity-custom-VJP statement does not resolve this.

Locations: `/private/tmp/marin-native-jax-oproj-full-cpu-backend/research/oproj/full_stage_a.md` lines 36-39.

Minimal correction: Name the exact flag and briefly state what its false and true values mean for these capture paths.

## Evidence boundary

The supplied prose explicitly separates local CPU validation and abstract tracing from full-target numerical acceptance, physical evidence, dispatch, performance, and adoption. No finding challenges the truth of source or test claims.

## exactinputSHA256

- `/private/tmp/marin-native-jax-oproj-full-cpu-backend/research/oproj/full_stage_a.md`: `7c342157dfb53a7ad72f6dddc69f63ec35fbac70b2e7059b1ffbe77da0654afb`
- `/private/tmp/native-jax-oproj-full-cpu-backend-preparation/report.md`: `e66a3c3176eaa7b217ae78242998cb0d9768329d72177d09b617c6cff56366a3`
- `/private/tmp/native-jax-oproj-full-cpu-backend-preparation/proposed-commit-message.txt`: `6ab266c926b8198633a824dd5e0e92d4f549a232afa0c93159088c7eab7f420a`
