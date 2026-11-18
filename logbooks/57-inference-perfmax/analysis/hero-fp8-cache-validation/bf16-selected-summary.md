# Inference performance campaign

Evidence: [s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-hero-bf16-01a1ed83](s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-hero-bf16-01a1ed83) ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fknotmeter%2Fcampaigns%2Fperfmax-hero-bf16-01a1ed83))

Throughput counts completed output tokens. Latencies are seconds. Each row is a separate trial.

| Trial | Runtime | Weights | GPU job | Sessions | GPUs | Output tokens/s/GPU | Change | p90 TTFT | p90 completion | AgentX | Checks | GPU-hours |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| [20261118T072255Z-4698a9d73b3e](s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-hero-bf16-01a1ed83/trials/20261118T072255Z-4698a9d73b3e/result.json) | [0ac9dcb157](https://github.com/marin-community/vllm/commit/0ac9dcb157ee86ccb447bc717259159c41ad456a) | bfloat16 | preparation-failed | 32 | 16 GB200 | unknown | — | unknown | unknown | no window | model-numerics: pending, loaded-weight-methods: pending | 0.000 |
| [20261118T072255Z-b78aac7ff367](s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-hero-bf16-01a1ed83/trials/20261118T072255Z-b78aac7ff367/result.json) | [0ac9dcb157](https://github.com/marin-community/vllm/commit/0ac9dcb157ee86ccb447bc717259159c41ad456a) | bfloat16 | preparation-failed | 64 | 16 GB200 | unknown | — | unknown | unknown | no window | model-numerics: pending, loaded-weight-methods: pending | 0.000 |
| [20261118T072838Z-3cca138ac87b](s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-hero-bf16-01a1ed83/trials/20261118T072838Z-3cca138ac87b/result.json) | [0ac9dcb157](https://github.com/marin-community/vllm/commit/0ac9dcb157ee86ccb447bc717259159c41ad456a) | bfloat16 | succeeded | 32 | 16 GB200 | 19.292 | -18.29% | 145.369 | 156.444 | valid | model-numerics: passed (provisional policy), loaded-weight-methods: passed | 23.500 |
| [20261118T073018Z-33071107ec92](s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-hero-bf16-01a1ed83/trials/20261118T073018Z-33071107ec92/result.json) | [0ac9dcb157](https://github.com/marin-community/vllm/commit/0ac9dcb157ee86ccb447bc717259159c41ad456a) | bfloat16 | succeeded | 64 | 16 GB200 | 23.610 | +0.00% | 276.382 | 282.529 | valid | model-numerics: passed (provisional policy), loaded-weight-methods: passed | 23.888 |
| [20261118T091911Z-31c846881afd](s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-hero-bf16-01a1ed83/trials/20261118T091911Z-31c846881afd/result.json) | [0ac9dcb157](https://github.com/marin-community/vllm/commit/0ac9dcb157ee86ccb447bc717259159c41ad456a) | bfloat16 | succeeded | 128 | 32 GB200 | 31.429 | +33.12% | 268.899 | 281.349 | valid | model-numerics: passed (provisional policy), loaded-weight-methods: passed | 50.406 |
| [20261118T111414Z-596d12bff98c](s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-hero-bf16-01a1ed83/trials/20261118T111414Z-596d12bff98c/result.json) | [0ac9dcb157](https://github.com/marin-community/vllm/commit/0ac9dcb157ee86ccb447bc717259159c41ad456a) | bfloat16 | failed | 128 | 32 GB200 | unknown | — | unknown | unknown | no window | actual-worker-numa: pending, model-numerics: pending, loaded-weight-methods: pending | 0.278 |
| [20261118T112726Z-ea23d4021e92](s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-hero-bf16-01a1ed83/trials/20261118T112726Z-ea23d4021e92/result.json) | [0ac9dcb157](https://github.com/marin-community/vllm/commit/0ac9dcb157ee86ccb447bc717259159c41ad456a) | bfloat16 | succeeded | 128 | 32 GB200 | 31.119 | +31.81% | 276.226 | 289.073 | valid | actual-worker-numa: passed, model-numerics: passed (provisional policy), loaded-weight-methods: passed | 51.146 |

Total recorded GPU-hours: 149.217.
Cost includes every retained Iris attempt, including setup, warmup, drain and failed attempts. Missing allocation timestamps remain unknown; queueing is excluded.


Trials with different controls form separate comparison groups. Their throughput changes are omitted.

Eligible for recommendation: 20261118T072838Z-3cca138ac87b, 20261118T073018Z-33071107ec92, 20261118T091911Z-31c846881afd, 20261118T112726Z-ea23d4021e92.

Recommended trial: 20261118T091911Z-31c846881afd. Settings and runtime: [s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-hero-bf16-01a1ed83/recommended.yaml](s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-hero-bf16-01a1ed83/recommended.yaml) ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fknotmeter%2Fcampaigns%2Fperfmax-hero-bf16-01a1ed83%2Frecommended.yaml)).

Serving checks do not qualify RL learning or trainer probability ratios.
