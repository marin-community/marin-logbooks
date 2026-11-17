import sys, re, pandas as pd
CJK = re.compile(r"[一-鿿぀-ヿ]+")
for f in sys.argv[1:]:
    fl = pd.read_parquet(f)
    h = fl[fl.F4_cjk_leak & fl.bench.isin(["ifbench","mmlu-pro","cruxeval","math500","gsm8k-0shot"])]
    for b, g in h.groupby("bench"):
        print("==", b, len(g))
        for r in g.sample(min(4, len(g)), random_state=0).itertuples():
            s = r.reasoning + r.text; m = CJK.search(s)
            print(f"   [{r.model}] runs={len(CJK.findall(s))} cjkchars={sum(map(len, CJK.findall(s)))} :: {s[max(0,m.start()-120):m.end()+80]!r}")
