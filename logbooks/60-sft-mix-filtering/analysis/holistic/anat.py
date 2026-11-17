import sys, pandas as pd, numpy as np
df = pd.read_pickle(sys.argv[1])
df = df[df.bench!="math500"]
def cat(r):
    if not r.long: return "short"
    if r.loop: return "long:loop"
    if not r.boxed: return "long:no-answer"
    return "long:answered-right" if r.correct else "long:answered-wrong"
df["cat"] = df.apply(cat, axis=1)
t = (df.groupby(["bench","arm","step"]).cat.value_counts(normalize=True).unstack().fillna(0)*100).round(1)
print(t.to_string())
# problem-level: does long-share concentrate on same problems across arms?
for bench in ["aime24","olympiadbench"]:
    p = df[(df.bench==bench)&(df.step=="s222")].groupby(["doc","arm"]).long.mean().unstack()
    print(bench, "problem-level long-share correlation across arms:\n", p.corr().round(2).to_string())
    acc = df[(df.bench==bench)&(df.step=="s222")].groupby(["doc","arm"]).correct.mean().unstack()
    print(" problems never solved by control:", int((acc.control==0).sum()), " their long share (ctrl):", round(p.control[acc.control==0].mean(),2), " problems always solved by ctrl:", int((acc.control==1).sum()), "long share:", round(p.control[acc.control==1].mean(),2))
