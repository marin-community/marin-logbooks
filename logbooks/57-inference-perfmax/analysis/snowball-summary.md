# Inference performance campaign

Evidence: [s3://marin-us-east-08a/marin/knotmeter/campaigns/perfmax-workflow-snowball](s3://marin-us-east-08a/marin/knotmeter/campaigns/perfmax-workflow-snowball) ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fknotmeter%2Fcampaigns%2Fperfmax-workflow-snowball))

Throughput counts completed output tokens. Latencies are seconds. Each row is a separate trial.

| Trial | Runtime | GPU job | Sessions | GPUs | Output tokens/s/GPU | Change | p90 TTFT | p90 completion | AgentX | Checks | GPU-hours |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| [20261117T015312Z-6ca63b0cc654](s3://marin-us-east-08a/marin/knotmeter/campaigns/perfmax-workflow-snowball/trials/20261117T015312Z-6ca63b0cc654/result.json) | [0ac9dcb157](https://github.com/marin-community/vllm/commit/0ac9dcb157ee86ccb447bc717259159c41ad456a) | submission-rejected | 8 | 8 H100 | unknown | — | unknown | unknown | no window; diagnostic | model-numerics: pending | 0.000 |
| [20261117T015550Z-e7ef55077e67](s3://marin-us-east-08a/marin/knotmeter/campaigns/perfmax-workflow-snowball/trials/20261117T015550Z-e7ef55077e67/result.json) | [0ac9dcb157](https://github.com/marin-community/vllm/commit/0ac9dcb157ee86ccb447bc717259159c41ad456a) | succeeded | 8 | 8 H100 | 11.922 | — | 4.904 | 20.337 | invalid; diagnostic | model-numerics: pending | 0.996 |
| [20261117T021916Z-32b1ee916242](s3://marin-us-east-08a/marin/knotmeter/campaigns/perfmax-workflow-snowball/trials/20261117T021916Z-32b1ee916242/result.json) | [0ac9dcb157](https://github.com/marin-community/vllm/commit/0ac9dcb157ee86ccb447bc717259159c41ad456a) | succeeded | 8 | 8 H100 | 11.733 | — | 2.850 | 20.125 | invalid; diagnostic | model-numerics: pending | 0.842 |

Total recorded GPU-hours: 1.838.
Cost includes every retained Iris attempt, including setup, warmup, drain and failed attempts. Missing allocation timestamps remain unknown; queueing is excluded.


Trials with different controls form separate comparison groups. Their throughput changes are omitted.

Eligible for recommendation: none.

No accepted recommendation selected. Passing serving checks does not establish RL learning quality.
