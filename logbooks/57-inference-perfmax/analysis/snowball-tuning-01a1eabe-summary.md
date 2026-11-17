# Inference performance campaign

Evidence: [s3://marin-us-east-08a/marin/knotmeter/campaigns/perfmax-snowball-tuning-01a1eabe](s3://marin-us-east-08a/marin/knotmeter/campaigns/perfmax-snowball-tuning-01a1eabe) ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fknotmeter%2Fcampaigns%2Fperfmax-snowball-tuning-01a1eabe))

Throughput counts completed output tokens. Latencies are seconds. Each row is a separate trial.

| Trial | Runtime | GPU job | Sessions | GPUs | Output tokens/s/GPU | Change | p90 TTFT | p90 completion | AgentX | Checks | GPU-hours |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| [20261117T182804Z-6603a552927f](s3://marin-us-east-08a/marin/knotmeter/campaigns/perfmax-snowball-tuning-01a1eabe/trials/20261117T182804Z-6603a552927f/result.json) | [0ac9dcb157](https://github.com/marin-community/vllm/commit/0ac9dcb157ee86ccb447bc717259159c41ad456a) | succeeded | 128 | 8 H100 | 124.396 | +0.00% | 103.331 | 297.616 | valid | model-numerics: passed (provisional policy) | 11.369 |
| [20261117T201915Z-0af609b36ad6](s3://marin-us-east-08a/marin/knotmeter/campaigns/perfmax-snowball-tuning-01a1eabe/trials/20261117T201915Z-0af609b36ad6/result.json) | [0ac9dcb157](https://github.com/marin-community/vllm/commit/0ac9dcb157ee86ccb447bc717259159c41ad456a) | succeeded | 64 | 8 H100 | 92.302 | -25.80% | 60.396 | 154.101 | valid | model-numerics: passed (provisional policy) | 12.739 |

Total recorded GPU-hours: 24.108.
Cost includes every retained Iris attempt, including setup, warmup, drain and failed attempts. Missing allocation timestamps remain unknown; queueing is excluded.


Eligible for recommendation: 20261117T182804Z-6603a552927f, 20261117T201915Z-0af609b36ad6.

Recommended trial: 20261117T182804Z-6603a552927f. Settings and runtime: [s3://marin-us-east-08a/marin/knotmeter/campaigns/perfmax-snowball-tuning-01a1eabe/recommended.yaml](s3://marin-us-east-08a/marin/knotmeter/campaigns/perfmax-snowball-tuning-01a1eabe/recommended.yaml) ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fknotmeter%2Fcampaigns%2Fperfmax-snowball-tuning-01a1eabe%2Frecommended.yaml)).

Serving checks do not qualify RL learning or trainer probability ratios.
