"""Build the label-stripped five-arm Panel-60 grading pool (run by the executor after the three arm runs land).

Usage: python3 -I build_blind_pool.py BEN_PROTOCOL OUT_DIR ARM=OUTPUT_DIR ... (exactly the five bound arms).
Writes OUT_DIR/pool.json (to the grader) and OUT_DIR/key.json (kept by the executor; never sent to the grader).
Per item, answers get letters A-E from a seeded shuffle. The pool holds only the item index, the prompt messages,
Ben's reference answer and the final answer text. A row that hit max_tokens (finish_reason length) is graded as is,
with no marker in the pool; its finish reason is recorded per arm and item in key.json. It drops arm names, served-model names, run names, request and
response metadata, and any old grades. The executor opens key.json only after the grader's 300 grades are saved.
"""
import hashlib, json, random, sys
from pathlib import Path

ARMS = ["A13", "A8a", "A3", "v0p1", "v0p1-sysdefault"]
SEED = 20261031
proto = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
dirs = dict(a.split("=", 1) for a in sys.argv[3:])
assert sorted(dirs) == sorted(ARMS), sorted(dirs)
rows = proto["identity_rows"]
assert len(rows) == 60
answers, served, finish = {}, {}, {}
for arm, d in dirs.items():
    recs = sorted((json.loads(p.read_text()) for p in Path(d, "answers").glob("*.json")), key=lambda r: r["index"])
    assert [r["id"] for r in recs] == [r["id"] for r in rows] and all(r["status_code"] == 200 for r in recs), arm
    msgs = [json.loads(r["response_raw"])["choices"][0] for r in recs]
    assert all(m["finish_reason"] in ("stop", "length") and isinstance(m["message"]["content"], str) for m in msgs), arm
    finish[arm] = {i: m["finish_reason"] for i, m in enumerate(msgs) if m["finish_reason"] != "stop"}
    answers[arm] = [m["message"]["content"] for m in msgs]
    served[arm] = {r["request"]["model"] for r in recs}
names = [n for s in served.values() for n in s]
rng = random.Random(SEED)
pool, key = [], {}
for i, row in enumerate(rows):
    order = ARMS[:]
    rng.shuffle(order)
    letters = "ABCDE"
    key[str(i)] = {letters[j]: arm for j, arm in enumerate(order)}
    pool.append({"index": i, "prompt": row["messages"][:-1], "reference": row["messages"][-1]["content"],
                 "answers": {letters[j]: answers[arm][i] for j, arm in enumerate(order)}})
text = json.dumps(pool, ensure_ascii=False)
leaks = [n for n in names + ARMS + ["sysdefault", "panel60"] if n in text and n not in json.dumps(rows, ensure_ascii=False)]
assert not leaks, f"label leak in pool text: {leaks}"
out.mkdir(parents=True, exist_ok=False)
(out / "pool.json").write_text(json.dumps(pool, indent=1, ensure_ascii=False) + "\n")
(out / "key.json").write_text(json.dumps(dict(seed=SEED, arms=ARMS, key=key, non_stop_finish=finish), indent=1) + "\n")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps({"items": len(pool), "answers": 5 * len(pool), "non_stop_rows": {a: len(f) for a, f in finish.items()}, "pool_sha256": sha(out / "pool.json"), "key_sha256": sha(out / "key.json")}))
