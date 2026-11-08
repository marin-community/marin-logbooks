# CPU build inputs and Iris packaging corrections

Eight build-tool wheels are pinned and locally cached for the proposed CPU
shutdown diagnostic. Their hashes and declared dependencies pass a metadata
check against the declared requirements for Python 3.12.12. Source review also found two Iris packaging problems:
the editable-build hook checks the wrong field, and the prepared shallow Git
copy reports a different Iris revision date from the full-history checkout.
The [evidence index](shutdown-candidate/build-inputs/source-audit.json) binds the
artifact hashes and source identity. The Marin shutdown source candidate
`2674b5d3975e5c1ecac05fb4310814c90ec736e1` is unchanged;
no package installation, real backend build, image build, runtime service,
numerical kernel, job or terminal experiment query occurred.

The diagnostic is intended to exercise coordinated telemetry finalization before
JAX shutdown in separate clean and injected-error CPU attempts. Historical
shutdown failures remain unexplained. These packaging findings do not establish
their initiating cause or qualify the repair for runtime use.

## Build-tool closure

The previous [source packaging record](shutdown-runtime-packaging.md) left build
backends outside the frozen runtime lock. Metadata resolution now selects
Hatchling 1.33.0 and uv-build 0.9.30, plus packaging 26.3, pathspec 1.1.1,
pluggy 1.7.0, tomlkit 0.15.2 and trove-classifiers 2026.9.21.13. Inspection of
the selected Hatchling wheel adds its editable-build requirement,
`editables~=0.3`, resolved to editables 0.6 while retaining the first seven pins.
The eight workspace build configurations declare no other build dependencies;
the inspected uv-build editable hook returns an empty dependency list.

The [final build-tool requirements](shutdown-candidate/build-inputs/editable-build-requirements.txt)
retain distribution hashes. The [download inventory](shutdown-candidate/build-inputs/wheel-acquisition-r02.json)
selects one exact wheel per package with its URL and verified SHA-256. Wheels
remain local; they were read as ZIP data, never installed or imported. The
[metadata audit](shutdown-candidate/build-inputs/wheel-metadata-audit.json) checks
names, versions, Python requirements, active declared dependencies and nominal
wheel tags. The uv-build wheel targets manylinux 2.17 x86_64; the other seven
have platform-independent Python tags. Actual OS/library compatibility is
unqualified because no base image has been selected.

The resolver warned that it would use an available Python 3.12.13 interpreter
instead of requested 3.12.12 for building dependencies. Source builds were
disabled. The independent metadata audit explicitly evaluates Python 3.12.12;
it does not demonstrate that this interpreter or any wheel can execute in the
future image. An earlier invocation failed at argument parsing because it
combined incompatible `--no-build` and `--only-binary` options. Both attempts
and the additional editable dependency resolution are retained. None installed
packages or changed the frozen runtime lock.

## Editable hook correction proposal

Hatchling passes `target_name="wheel"` and `version="editable"` to Iris's build
hook. Candidate `lib/iris/hatch_build.py` tests the target name against
`"editable"`, so its intended early return is missed. Three controls extract and execute the original hook method
from its abstract syntax tree (AST), reproducing stamp creation for editable, ordinary wheel and sdist
inputs. Three controls for a proposed `version == "editable"` guard avoid the
editable stamp while preserving ordinary-wheel and sdist stamp creation. Revision lookup and
temporary-directory allocation are faked; no backend or package initializer ran.

The [one-line patch](shutdown-candidate/build-inputs/iris-editable-guard.patch)
passes `git apply --check` against the prepared source copy but is not applied.
Actual built wheel contents and installed import precedence remain untested.
This corrects the previous packaging review's assumption that the current hook
skips stamping during editable builds. The original publication remains intact.

## Shallow-history date mismatch

Iris derives its revision date from the last commit touching its source tree
when neither `IRIS_REVISION_DATE` nor the baked `iris/_build_info.py` stamp
is available. A read-only
local Git query at the exact candidate gives `2026-10-29`, commit
`8a92f90382559206b03ead1afbbd32c35f68fcff`, in the full-history checkout. The same
query in the prepared depth-one copy gives the shallow boundary commit and
`2026-11-08`. Matching source bytes therefore do not preserve this derived date.
The earlier eight-file preflight did not test path-history-derived identity.

A future image recipe must either retain sufficient Git history or bind the
verified Iris date explicitly, with its source provenance. Neither option has
been selected or applied. This source date is separate from the still-unverified
deployed controller revision. The [input proposal](shutdown-candidate/build-inputs/build-input-proposal.json)
also retains the unresolved base image, OS, installer, 160 runtime-wheel inputs
and independent controller/coordinator evidence acquisition. The latter must
join task attempts to endpoint registration and removal events, including
removal reasons; final endpoint absence cannot certify application cleanup.
Image construction, real runtime initialization and diagnostic jobs require
separate authorization. Future diagnostic attempts require ONLINE W&B tracking
and independent task/native completion evidence. No GPU allocation may be
voluntarily released without user consent.
