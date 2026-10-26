# Measurement archive for Marin #10682

Contribution: https://github.com/marin-community/marin/pull/10682
Marin source: 1545fb4898912be01051e82e1796906d667b78ab
Archive guidance revision: f54b3aa0e6ee533e988d2b97e9db0d1523f934dd

The measurements compare margin-form fp32 trainer arithmetic with the existing TE/bf16 path.
Production training means logprobs with gradients plus entropy without gradients; scoring means
logprobs without gradients. The recorded32K H100 training pass is29.56ms versus67.41ms, with
9.27GiB versus9.56GiB peak allocated memory above inputs. Scoring is7.65ms versus7.42ms.
The ten-shape run records one forward/one backward binary per output specialization.

Saved Qwen2.5-0.5B PPO replay agrees to7.45e-9 loss and0.019% gradient norm, with identical initial
parameters and forward logits. Main's active-token exact-zero entropy rate is19.97%; stable is0%.
Two main/two branch three-step CatCount runs average8.54–8.63s and8.11s per step respectively.
These small-run timings include startup effects and do not establish steady-state throughput.

The production gate accepts about3% scoring overhead and lower training time/memory. Isolated
logprob tuning is a separate follow-up. The unchanged programs are archived here with their full
Marin source tree so their interfaces remain reproducible; raw/private evidence stays in its
existing controlled artifact stores. See README.md for entry points and access limitations.
