import sys; sys.path.insert(0, sys.argv[2]); from load import runs
R = runs(sys.argv[1])
c = R[("control","s222","humanevalplus")].set_index("doc_id")
for arm in ["cap32k","cap32k-giveup"]:
    a = R[(arm,"s222","humanevalplus")].set_index("doc_id")
    nocode = ~a.output.str.contains("```")
    ok = a.correct.astype(bool); cok = c.correct.astype(bool).reindex(a.index)
    lost = ok.eq(False) & cok
    print(arm, "no-codeblock outputs:", int(nocode.sum()), "of which wrong:", int((nocode & ~ok).sum()), "| lost items with no codeblock:", int((lost & nocode).sum()), "of", int(lost.sum()))
    print("  acc on outputs WITH codeblock:", round(ok[~nocode].mean(),3), "control:", round(cok[~nocode].mean(),3))
    for o in a.output[nocode].head(3): print("   >>", repr(o[:250]))
