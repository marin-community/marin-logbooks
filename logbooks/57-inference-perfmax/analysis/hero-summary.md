# Inference performance campaign

Evidence: [s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-workflow-hero](s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-workflow-hero) ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fknotmeter%2Fcampaigns%2Fperfmax-workflow-hero))

Throughput counts completed output tokens. Latencies are seconds. Each row is a separate trial.

| Trial | Runtime | GPU job | Sessions | GPUs | Output tokens/s/GPU | Change | p90 TTFT | p90 completion | AgentX | Checks | GPU-hours |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| [20261117T015313Z-f85a8d257772](s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-workflow-hero/trials/20261117T015313Z-f85a8d257772/result.json) | [0ac9dcb157](https://github.com/marin-community/vllm/commit/0ac9dcb157ee86ccb447bc717259159c41ad456a) | submission-rejected | 8 | 16 GB200 | unknown | — | unknown | unknown | no window; diagnostic | model-numerics: pending | 0.000 |
| [20261117T015550Z-b09439ebe1fc](s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-workflow-hero/trials/20261117T015550Z-b09439ebe1fc/result.json) | [0ac9dcb157](https://github.com/marin-community/vllm/commit/0ac9dcb157ee86ccb447bc717259159c41ad456a) | failed | 8 | 16 GB200 | unknown | — | unknown | unknown | no window; diagnostic | model-numerics: pending | 0.485 |
| [20261117T021522Z-21bf94f72725](s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-workflow-hero/trials/20261117T021522Z-21bf94f72725/result.json) | [0ac9dcb157](https://github.com/marin-community/vllm/commit/0ac9dcb157ee86ccb447bc717259159c41ad456a) | succeeded | 8 | 16 GB200 | 0.377 | — | 109.042 | 109.042 | invalid; diagnostic | model-numerics: pending | 4.548 |

Total recorded GPU-hours: 5.033.
Cost includes every retained Iris attempt, including setup, warmup, drain and failed attempts. Missing allocation timestamps remain unknown; queueing is excluded.


Trials with different controls form separate comparison groups. Their throughput changes are omitted.

Eligible for recommendation: none.

No accepted recommendation selected. Passing serving checks does not establish RL learning quality.
