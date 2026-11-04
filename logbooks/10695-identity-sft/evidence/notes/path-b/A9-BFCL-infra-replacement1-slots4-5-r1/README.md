# A9 BFCL one whole infra replacement each: original slots 4 and 5

Namespace-only successor of the already submitted fresh-recovery A9 BFCL packet. Original slot 4 has 122/123 scored and slot 5 has 119/123 scored, with EnvironmentStartTimeoutError; preserve all original rewards and NULLs. Original slot 3 is valid 27/123 and must never be rerun. No partial arm mean or CI.

API: `prepare("A9", 4)` or `prepare("A9", 5)`, then `submit(batch, client)` in the SAME isolated process for each prepare/submit pair. Exact model YAML, HF, evaluator, policy and prepare/submit implementation are unchanged. Verify current orders, fresh output namespaces, ordinary actual-HF/write/native guards, and arm one notifier per actual submission. ONE R1 per authorized logical slot; no second replacement.

Original slot 1 diagnostic cohort remains unchanged; a replacement never substitutes for that diagnostic. No Iris/model/storage calls by author.
