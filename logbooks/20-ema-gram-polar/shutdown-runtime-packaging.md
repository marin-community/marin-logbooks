# CPU shutdown diagnostic packaging preparation

An isolated source checkout and diagnostic harness are prepared to test whether
four CPU processes finish telemetry, exchange completion records and shut down
JAX distributed services in that order. Historical training failures included
a three-of-four shutdown barrier timeout; their initiating cause remains unknown. The checkout retains candidate commit
`2674b5d3975e5c1ecac05fb4310814c90ec736e1` and its Git objects without the
candidate worktree's shared virtual environment. Five harness files (worker, diagnostic, validator, trace adapter and manifest)
match the published [CPU entrypoint](shutdown-cpu-entrypoint.md). Separately,
read-only preflight binds eight candidate library files to their Git commit and
working bytes. It reports
`launch_ready=false`. No runtime image was built, package installed, service
initialized or job submitted. Frozen training source and scientific
classifications remain unchanged.

The [artifact index](shutdown-candidate/runtime-packaging/source-audit.json)
records hashes for preparation scripts, source and dependency inventories,
scoped local checks and an evidence collection contract in the same directory. The 121 MB source tree stays local;
the archive records its commit and tree identity rather than duplicating it.

## Source and dependency preparation

The source preparer creates a separate shallow Git clone through local Git
transport, checks the requested commit, and removes its source remote. It
rejects dirty tracked source and existing destinations. Three temporary Git
fixture checks pass. The actual candidate copy has a self-contained `.git`
directory, no external object alternates, no copied `.venv` and a clean tracked
status. Its tree has 7,540 entries, four internal symlinks, no submodules and no
matches for the Git LFS pointer header. The tracked attributes file declares
only a generated dashboard file; it contains no checkout filter. This audit
covers source materialization, not package installation or runtime imports.

A frozen, offline `uv export` left the candidate's `uv.lock` unchanged. A cached
split selects 160 registry requirements for the proposed Linux x86_64,
CPython 3.12.12 target and separates eight workspace packages that must be
installed from the source checkout. The registry output retains exact versions,
artifact hashes and platform conditions. Selected versions include JAX/JAXlib
0.11.1 and W&B 0.26.0. Target selection does not establish compatible wheel
availability or a working runtime. The Python patch version is a packaging
proposal, not an observed execution environment.

Seven distinct splitter checks have scoped passing evidence. Six ran before a
correction to distribution-name normalization; one new check with three spelling
cases and one affected duplicate check passed afterward. Five unaffected checks
were reused. The cached real split was not repeated. Together with the three
Git fixture checks, this is ten distinct local checks, not a fresh full suite.
Source lint passes. The new context's read-only preflight is recorded separately.

The dependency export omits build backends: seven workspace packages declare
unversioned `hatchling`, and Levanter declares `uv_build>=0.7.19,<0.10.0`.
Neither backend appears in the root runtime lock. Exact backend wheels and their
dependencies, the installer, base image and OS packages still need pinned
artifacts. Iris's build hook imports `iris.version`; a future package build
therefore executes package code and remains outside these source checks.
No shared environment was synchronized or modified.

## Independent runtime evidence

The [evidence contract](shutdown-candidate/runtime-packaging/evidence-collection-contract.json)
requires separate source/image, controller, task-attempt, W&B history from online runs,
coordinator-registration and local lifecycle records. These records must join
on fresh execution and attempt identities. No collector has run.

Source inspection found a limit in the endpoint registry API: `ListEndpoints`
returns current live entries but omits attempt numbers and UIDs, registration
and lease timestamps, deletion reasons and an event cursor. The inspected
`EndpointService` exposes registration, unregistration and listing; its local
projection callbacks are not a remote replay API. Registration also renews
leases and updates the stored registration timestamp. An idempotent successful
unregister response does not establish that a live entry was removed.

A final absent endpoint can result from explicit unregister, expiry,
replacement or other cleanup. It cannot by itself prove the application's
cleanup path or coordinator lifetime. Independent retained lifecycle evidence
must therefore be arranged before any future authorized attempt. If the
required lifecycle evidence is unavailable, explicit application cleanup and
coordinator lifetime remain unqualified even if task exit status is known. This source finding does not identify the historical failure's
initiating cause or establish the deployed controller's behavior.

Tracking entity, CPU placement, immutable image and build inputs, deployed
controller identity and evidence acquisition remain unresolved. The proposed
300-second budget bounds each coordinator wait for all ranks to publish or
finish reading completion records. Expiry makes the local finalizer fail before
its explicit JAX shutdown callback; real cleanup on that path remains unknown.
This proposed communication-health budget is unchanged and unmeasured. No queue,
task or wrapper deadline is introduced. CPU preparation does not resolve later
accelerator retention. New runtime execution still requires explicit authorization.
