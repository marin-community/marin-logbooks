# Callback compatibility preparation revision 2

This revision addresses the two P2 findings in the fresh scientific review.
The original source/config/command/prose freeze, handoff, review, and CPU
receipts remain preserved. The findings were source-level release blockers,
not observed numerical failures of that retained CPU run.

Each transport and composition execution now receives its own fresh actual
Collector. Its finish runs only after that execution's result and effects
barriers. Every execution must independently provide the global X/G and all
four canonical local identities. Transport minima remain X=2/G=2 per identity;
each rematerialized composition execution requires X=2/G=1, so the two-call
minima are not weakened. Repeated/rematerialized records remain accepted only
when identical. All ten capture shape/dtype/byte identities are explicitly
compared across the composition executions, independently of count equality.

The final positive CPU v2 report passed both Auto/Auto and Explicit/Explicit
2x2 meshes, six real lower/compile operations and eight compiled executions.
Transport counts were X=2/G=2 for every global/local identity. Each individual
composition execution had X=2/G=1 for every global/local identity. All capture
oracles, cross-execution byte comparisons, and loss plus three floating-input
gradients matched exactly. The unchanged finite BF16 fixtures and actual
dynamic scan indices [0,1] were used. Twelve complete tiny HLO files and JSON
total 414,506 bytes. No arrays/checkpoints/profiles/models were written.

The CPU-only missing-record negative runs the same actual worker/compiled
composition with deliberate receive-side faults. The first composition has an
extra identical G/ordinal-3 observation, count 2. The second composition omits
that identity entirely while retaining the other nine identities. Its own
collector finish rejects KeyError(('G',3)); the earlier duplicate cannot make
it pass. This negative uses the normal runtime/source guards and performs no
TPU execution.

The frozen direct commands now invoke a minimal stdlib supervisor. It starts
the worker in its own process group, begins its wall clock before worker
creation, and enforces an external 300-second wait across all worker imports,
native compilation, execution, and barriers. On timeout it sends SIGKILL to
the owned group, reaps the worker, writes the failure receipt, and exits 124.
The internal SIGALRM is now explicitly described as diagnostics only. Root
must additionally pin the reviewed Iris task timeout. The positive worker
completed under this supervisor in 8.09 seconds. A bounded 0.25-second timeout
control against a sleeping Python worker returned 124, recorded SIGKILL exit
-9, and elapsed 0.253 seconds. No recovery path was used.

Normal offline infra/pre-commit.py passed all seven research files using the
existing cached tools. The original source A checkout remains clean at
32f42f9a3601258123d316cc03c25b05add7a9c2. The capture candidate is unchanged:
Collector and make_tap ASTs equal source A, and emit differs only in the inner
callback transport Boolean. Original runtime/environment/scientific-source
hashes, mesh/ids/index0/custom-VJP checks, and fixture criteria remain intact.
No original failed A or parent archive was edited.

This remains a private preparation. No commit, publication, network, storage,
client, job operation, or TPU execution occurred. Root must obtain a targeted
fresh scientific re-review, release exact files before any commit/publication,
publish the new immutable source/config/command, and record the dispatch
release before deciding whether to run the tiny TPU gate. CPU/negative/source
checks do not prove TPU compatibility or full-model numerical cause, parity,
trajectory, performance, or adoption.
