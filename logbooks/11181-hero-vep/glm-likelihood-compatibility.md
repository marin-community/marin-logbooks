# GLM likelihood compatibility check

GLM-5.3 is reachable and returns prompt log probabilities, but the tested
route failed stability checks. No GLM panel run or valid VEP result was produced.
The comparison remains restricted to general language models; DNA specialists
are out of scope.

## Tokenizer

GLM and Mochi use different tokenizers. In the tested GLM service, ` A` is
one token (362), ` B` is one token (425), and `Answer` is 16127. Mochi uses
362, 426, and 16533 respectively. The complete `Answer: A`/`Answer: B` strings
preserve the query prefix plus the corresponding one-token candidate.
GLM classification was not evaluated.

GLM tokenization of an empty input returns no BOS token. The initial DNA token
has no prompt log probability, and it is identical between the tested reference
and alternate. For a null-prefix ref/alt contrast, that common unscored token
can cancel; require that identity explicitly rather than silently dropping
different tokens. The healthy prefix is nine tokens and remains an exact prefix
of the full DNA input. All returned DNA token strings and native probabilities
align with the actual input, without inclusion of generated output.

## Stability failure

One existing autosomal pilot variant was used. Four initial likelihood requests
covered reference and alternate with null and healthy prefixes. HTTP success
and token alignment passed, but the identical shared prefixes differed by up
to 9.802 and 10.728 log-probability units per token respectively.

Four further requests compared identical inputs and zero versus one output
token. The zero/one-output pair matched exactly; two identical one-output
requests differed by up to 9.955 units per prompt token. This does not isolate
the source of the variation.

A final four-request probe supplied the documented completion-session affinity
key. Reference repeats still differed by up to 5.470 units per token, while
alternate repeats matched; the shared ref/alt prefix differed by up to 10.799.
Worker identity was not exposed in the returned headers, so same-worker routing
was not established. Do not attribute the issue to a particular numerical kernel
or model checkpoint from these observations. Do not hide it by subtracting
only a selected suffix or report these values as model-quality evidence.

## Execution and remaining blocker

Twelve tiny inference requests and ten tokenizer/metadata requests returned
HTTP 200. Probes used the approved low-priority service tier, one request at a
time and one-second pacing; each CPU job had one CPU, 2 GB RAM, a ten-minute cap
and no automatic failure retries. One earlier dependency-setup failure made
zero service requests. All four task-owned jobs are terminal. No new GPU
deployment or serving configuration change occurred.

The small diagnostic scripts were private, and were not committed before
execution; this is a research-recording deviation, not a frozen benchmark run.
Raw responses, deployment/access details, scripts and three downloaded output
archives with checksums are retained privately. The sanitized numerical audit
is `glm-likelihood-compatibility.json`.

The next prerequisite is a stable, validated GLM prompt-likelihood path, ideally
with service-owner guidance and identifiable routing. Tokenizer compatibility
is understood; likelihood validity remains the blocker. No GLM panel or reserved
300-variant Mochi validation has launched.
