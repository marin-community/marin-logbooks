import sys, pandas as pd
t = pd.read_parquet(sys.argv[1], columns=["model","bench","text"])
for m, b in [("sft-10.05","gsm8k-0shot"),("control-s222","gsm8k-0shot"),("sft-10.05","ifbench"),("control-s222","humanevalplus"),("sft-10.05","triviaqa")]:
    s = t[(t.model==m)&(t.bench==b)].sample(6, random_state=1)
    print("=====", m, b)
    for x in s.text: print("   >>", repr(x[:260]))
