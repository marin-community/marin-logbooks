# Contextless prose read

Contextless finished-prose read of the three supplied files only. No source audit, runtime inspection, network access, edits to inputs, or publication.

Two wording findings; no source or runtime correctness judgment.

## W1: Targeted review correction

File: `/private/tmp/marin-native-jax-callback-cpu-backend/research/oproj/callback_compatibility.md`

> This revision records complete per-execution captures and requires the external supervisor in both prepared CPU and target commands.

The unqualified "This revision" reads as the current tpu,cpu preparation, although the report says per-execution capture checks and the supervisor were already present in ed17 and are unchanged. The preceding "first preparation" is similarly undated. A future reader cannot reliably place these corrections before the new ambient-backend change.

Suggested wording:

> The earlier ed17 revision added complete per-execution captures and required the external supervisor in both prepared CPU and target commands. Those corrections and their controls are retained unchanged in this tpu,cpu preparation.

Likewise label the "New-fixture development" paragraph as earlier fixture history if that is the intended chronology.

## W2: Worker delta and AST comparison

File: `/private/tmp/native-jax-callback-cpu-backend-preparation-report.md`

> AST comparison proves every worker statement outside provenance is unchanged from ed17.

The preceding paragraph identifies both provenance changes and new fail-closed backend checks. "Outside provenance" leaves it unclear whether backend validation is included in that category, so the two delta descriptions appear inconsistent when read alone.

Suggested wording:

> The source audit confirms that the changes are limited to provenance and backend validation; the fixture execution logic is unchanged from ed17.

If preserving the AST-specific claim, name the exact audited block instead of "provenance"; this read did not inspect or validate the audit.

## Other checks

- The proposed title starts with the imperative "Prepare" and states the concrete change.
- The texts distinguish CPU results and installed-source reasoning from actual TPU success, original-environment fidelity, and full-model acceptance. No wording issue found in those boundaries.
- The matching default and explicit TPU device lists describe checks within a target execution; the prose does not claim identical physical chips across runs.
- The release and authorization boundary is repeated, but no additional approval or process is requested by this read.

## Input SHA-256

- `/private/tmp/native-jax-callback-cpu-backend-commit-message.txt`: `734d4703d6e6aabc57fa5b75c73fbc3baebae5f762c8de685d30c505e68203e8`
- `/private/tmp/marin-native-jax-callback-cpu-backend/research/oproj/callback_compatibility.md`: `e19c3012f41a32bec4b76d0970053edb68188a0dfc81e9b8901ba7b367e1a530`
- `/private/tmp/native-jax-callback-cpu-backend-preparation-report.md`: `3958e91728c30bdf4b4f70d901e25cd738069601ff751ee6544acecc11783b0f`
