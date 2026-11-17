import sys, re, pandas as pd
t = pd.read_parquet(sys.argv[1], columns=["model","bench","doc","prompt","text"])
o = t[(t.bench=="olympiadbench")]
print("olympiad prompts mentioning answer/solution of earlier parts:", o.drop_duplicates("doc").prompt.str.contains(r"(?i)(answer|solution)\s*(to|for|of)?\s*(the )?(previous|part|question)|Final answer|context", regex=True).mean())
p = o[o.doc.str.startswith("65a2241d")].prompt.iloc[0]
print(len(p)); print(p[:1500]); print("...", p[-1500:])
