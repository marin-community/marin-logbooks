# CD16-T2 panel binding: draft read passes

The draft arm files and interactive runner pass the early binding read. This is not a submit CLEAR: the candidate export, export-audit/readback receipts and actual template binding are still pending.

- Protocol: `a2f4e2afff0397c5eb736a7e1a051900b155711c5eb747d4d3e864ed73ea3141`.
- Serving: `605babe13e914157952e41557ac3f8a8c5059236fa236ab3786926b261a3d5e0`.
- Builder: `84c0651687ddec94f822e66d528817429d5c2420dd66c8bff2587a42fdbd07f4`.
- Interactive runner: `5fcce77b92dfb54269d279b0cfa8ed926939f51196df0e9898b289257a5a53d6`, byte-identical to the original copied runner. The sole difference from the previously reviewed production runner is the inference child's priority changing back to interactive, matching current orders.

An independent rebuild reproduces all four arms' protocol/serving files byte-for-byte, including the three existing arms. The new protocol changes only the served model and run label. All 60 rows preserve canonical SHA256 `0fafe46fb43cc8ec977fb84ba3ee67e932472644c64bd3bebfab3b841a160a10`; serving differs from the copied original only in model name, location and revision. Sampling, scoring, templates, resources and duration are not otherwise changed by this draft.

The model path targets the reviewed training plan's `train-r1/hf/step-15` export, revision unset. No export is inferred to exist or pass from that path. Before the actual-export read, bind its trained audit and regional readback hashes and confirm the plain source chat template SHA256 `b3f20bc21498407f2815dd1a6c26150192926ccf04f6324ca40a1209bff7d1de` without injected identity. The orchestrator submission must also use interactive priority. Runtime remains the previously reviewed baseline plus credential backport; no runtime re-review is implied by the arm builder alone.
