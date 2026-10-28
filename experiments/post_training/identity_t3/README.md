# CD16-T3 sampling-only training

T3 reuses the1,089 T2 native rows and all45 original cache files. The only
scientific change is sampling: old identity1.2, new v2 identity4.5, tools3.0,
third-person controls6.0, preservation1.0 per row. Component order is unchanged;
weights are264/547/504/468/324. Seed310, feistel/block2048, binary masks,
16updates, batch8, lr2e-6 and all precision/model/export implementation remain
unchanged. It starts fresh from v0.1, not from T2 weights.

The logical input packet is4f320bf5226777bcf173c0e7c52f14bc02e28f329ee12d9ec0f53cc0b6baee7e,
prepared by `identity_t2.freeze_sampling` at fbd7991bf8646daf26bcf4a5149974260579c3e6.
The first128 native examples contain28new identity,25tools,19new controls,
16old identity and40preservation sightings. Identity2912/47347=6.150337%;
tools1892/47347=3.996029%; controls11041/47347=23.319323%. These are supervised
token shares, not gradient shares. Tools25, fiction0 and the control share are
owner-accepted pre-results risks; no seed search or further weight changes.
History remains zero retained new rows.

Sequence: independent logical CLEAR; `identity_t2.relocate` with the required
approved logical packet hash to an exclusive RNO root; full readback and
native128/all1089 validity replay; `python -m experiments.post_training.identity_t3.launch`
with explicit inputs (plan only by default); combined independent CLEAR; then
one `--run` invocation with the exact clean pin/plan/freeze/orders/review hashes,
fresh R8/namespace/named-capacity guards, interactive priority, explicit timeout
and zero retries. Use client TMPDIR on/mnt/storage. No GPU is held for review.

The moved training adapter retains the original actual128 prebuffer audit,
per-update token/mask/query/key fingerprints,502FP32master audits, zero-drop
checks and native-before-HF export. Original T2a928 remains immutable in its
separate worktree; no compatibility shim or model/library patch is used.

After regional trained-export readback: production-exception gate, interactive
screen/math/BFCL, and Vizcacha's panel60 blind-graded together with T2. The six
ship lines are unchanged. No OPD follows; full-suite authority is conditional
on the first five lines passing, not on gate or controller success.
