# Full Stage A transport and IO review

Conditional pass: no actionable transport/IO code blocker found for the exact final local artifacts below. Publication, scientific review, post-publication capture and root dispatch release remain mandatory. This review does not authorize dispatch.

Source: `bef5c25fd34145fa4ec089594daee4e3c8bf1113`; S2 parent: `0be2ab8574125f932b18ab8ec5d04a564b432be5`.

Job: `/dlwh/levanter-native-jax-v5p-oproj-full-cpu-20261024-prep1`.

## Independent checks

- Read exact transport, validation, bootstrap, supervisor, source freeze, prose delta, control receipt, and applicable Iris output/secrets policy.
- Current source is clean at the exact bef5c25 commit with only child.py and full_stage_a.md changed from S2; 24 pinned source bytes match current command and bundle.
- Existing request.pb parsed with generated protobuf schemas only; MessageToDict without bundle equals saved proof; embedded bundle bytes equal saved source-bundle.zip.
- ZIP has 4717 unique safe relative paths, no banned credential/cache/local-build prefixes or symlinks, and exact size/hash inventory for every entry.
- Packed launcher statically decoded without execution and matches bootstrap.py bytes; embedded configuration and supervisor match their files exactly.
- Wire specifies v5p-8 with four TPU chips, 32000 millicores, 274877906944 memory bytes, 68719476736 disk bytes, one replica, required us-central1/us-central1-a, 120s scheduling timeout, 1020s execution timeout, ERROR existing-job policy, all three retry fields zero, and empty secret map.
- ISOLATE prevents ambient forwarded application secrets. Wire ordinary environment contains the three exact scientific settings plus documented SDK defaults and provenance.
- Release checks, immutable command/source/bootstrap/config/supervisor hashes, exact clean frozen descendant/two-file delta, proof binding and exclusive local claim all precede open_iris_client. Any claim persists through ambiguous RPC outcome; no automatic recovery is authorized.
- Bootstrap runs diagnostics outside IRIS_OUTPUT_DIR, uses unchanged retention function, checks symlink/special files and both pre/post-receipt bounds, then copies once. Worker nonzero/timeout status remains nonzero; retention rejection writes only bounded failure metadata.
- Supervisor starts a new session and kills the owned process group on its 900s deadline. Diagnostic and both synchronous scientific child processes inherit that group in the inspected source.
- Existing IO control receipt covers success, worker failure, timeout, per-file oversize, total oversize, disallowed array filename and symlink rejection before Iris copy. Existing supervisor receipt shows 0.25s timeout, worker -9, elapsed 0.254s.
- No root-dispatch-release.json, submission-claim.json or submission-state.json existed during review.
- Final transport-freeze.json matches every artifact hash. SDK runtime receipt uses query-scale-rounding Python with explicit PYTHONPATH pointing to this source worktree Iris/rigging/fray; all seven recorded module files independently match their hashes. Intended root submit uses the same executable/environment and remains unexecuted.

## Evidence limits

- Did not execute scientific/JAX/TPU code, SDK capture or bundle creation; did not construct network clients, authenticate, contact network/storage/jobs, read secret values, publish, commit or edit source.
- Existing control receipts were inspected rather than rerun. Timeout receipt exercises a sleeping leader; descendant cleanup follows statically from same-group child creation and killpg, without a separate live descendant receipt.
- SDK can retry the same logical launch RPC. Exclusive local claim plus server ERROR policy is a one-logical-submission fence, not an exactly-one-RPC guarantee.
- Staging validation bounds retained files, not guaranteed archive upload. Cancellation, preemption, worker loss or Iris capture failure can make archives unavailable. Setup time and queue delay can shorten the useful worker window before the Iris deadline.
- Current capture is post-local-commit and pre-publication. Public source/archive verification and post-publication recapture remain mandatory release work.
- Scientific CPU/abstract/negative evidence remains on config81f009; config85e075 changes only the documentation hash according to supplied prose-revision proof. No full-target numerical acceptance is claimed.
- No blanket audit of repository contents for secrets is claimed; checks cover empty wire secret map, ISOLATE behavior and forbidden bundle paths.
- RemoteClusterClient network constructor is bypassed by the recorded capture. The IrisClient facade constructor only stores the supplied stub-backed cluster and controller; its construction is not a network action.

## Remaining release conditions

- Separate scientific review passes for this exact source/configuration.
- Root verifies published source bef5c25 and exact archive, then recaptures actual SDK request from clean source after publication.
- Root release binds exact reviewed command/wrapper/config/bootstrap/job identity, final request-validation hash, source and archive commits, and explicit dispatch_authorized=true.
- Use the reviewed executable and explicit source-worktree PYTHONPATH recorded in SDK-runtime-provenance.json for post-publication recapture and any separately authorized submit; those module file hashes match the reviewed source.
- Any future dispatch/monitor window remains separately authorized and owned by root; preserve claim after any ambiguous result and inspect instead of resubmitting.

## Exact reviewed artifacts

| File | SHA-256 | Bytes |
| --- | --- | ---: |
| command.json | `ff042957283cfcaa169f222cf3975f230ae78be795dd2bcabc1e0283f3bb6cb5` | 16972 |
| config.json | `85e075747dcea4156c4d73a01ba41aa9d9d604f70415b682e23ad81ff5a0915b` | 55141 |
| bootstrap.template.py | `53f21798a555991d9b8b634aad232cfbce6a8477456a16806335ed9f590d3236` | 5366 |
| bootstrap.py | `3c1392b894e5da62812ecce2cdd9787812b7b696a833fd4f225ba878bb5c6818` | 64195 |
| worker_supervisor.py | `6d3fef98bd91533ad385e12479b733834ac9175f786c3e510c6cf53afa812132` | 1876 |
| submit_once.py | `1bd2b2fbdcf8c347040938cd340956a401fc0e9416b1acc420f58f224a3cd4c2` | 7726 |
| validate_request.py | `5cb6af42411b25e75ddf2d4b2d7dde2ba2efefa6cf775a08fa99ead0ffec8185` | 5266 |
| validate_io.py | `dd80e07f1e6715eba3970e9717599fc86b891095c4368b696c8b770d0e1ac860` | 4731 |
| request-validation.json | `df27b4aadc839f79adecbcfeffeed8f571f46f4d0983768a1f235f467440bccd` | 901381 |
| request.pb | `17dcc247fa527bd218b403fb9ee7b35d1103faaa3df54e93cae709bb02e971ec` | 16674327 |
| source-bundle.zip | `4e7c445fc87aaab20ebfe9bab7762f51f7c4c798561eb4a83433e0a654d4c6f5` | 16661016 |
| bootstrap-io-controls.json | `c37e9dbfabd7b7cc3d34a10b1208073a86a14750a2c758690785cdeec023cd74` | 1658 |
| supervisor-timeout-control.json | `a1394addd9786dc0e1bf4528040d807d59af8bfe747139771131b636f9be91fa` | 311 |
| transport-precommit.log | `c891b77212f254c34abdfe59f51289f32d890c09ba4b359aee63db4d2a0cf5af` | 563 |
| SDK-runtime-provenance.json | `a3a5375de9a067ddf74741d58d753559bf14901bb7074ed8d33883512510444f` | 2749 |
| transport-freeze.json | `4cf1841d4a929adc67945e5e6c6a344ff290fe4b2d1b7fb4e7649c3378404a39` | 2011 |

## Context evidence hashes

- `/private/tmp/native-jax-oproj-full-cpu-backend-preparation/source-commit-freeze.json`: `ca770c4cbc020d992ac6f23becc5f772afd1a928c33cfba6bda315ff16e28505`
- `/private/tmp/native-jax-oproj-full-cpu-backend-preparation/prose-revision-proof.json`: `c1827186fe044869d2c0cd4f0bab241fa651b7b05aa146c1f3ebe04a0d463124`
- `/private/tmp/native-jax-oproj-full-cpu-backend-root-local-commit-proof.json`: `d83294d5adb54f8f9bd231aeb68572449176e2099970158b09d435a5270d4b7a`
