"""Build panel-60 protocol/serving files for the Path B candidates from Ben's gist copies (same method as ../build_arms.py).

Usage: python3 build_arms_pathb.py BEN_GIST_DIR OUT_DIR. Only served_model/run (protocol) and name/location/revision
(serving) change; the 60 identity_rows are asserted equal to Ben's provenance rows_canonical_sha256.
"""
import copy, hashlib, json, os, sys

G, A = sys.argv[1], sys.argv[2]
proto = json.load(open(f"{G}/protocol.json"))
serve = json.load(open(f"{G}/serving.json"))
prov = json.load(open(f"{G}/provenance.json"))
canon = lambda rows: hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
assert canon(proto["identity_rows"]) == prov["rows_canonical_sha256"]
BASE = "s3://marin-rno2a/marin/checkpoints/karan/snowball-identity-pathb"
for arm in ("A13", "A8a", "A3"):
    name = f"snowball-pathb-{arm.lower()}-u36-panel60"
    p = copy.deepcopy(proto); p["served_model"] = name; p["run"] = f"panel60-pathb-{arm.lower()}-u36-20261031"
    s = copy.deepcopy(serve); s["model"].update(name=name, location=f"{BASE}/{arm}-u36", revision=None)
    assert canon(p["identity_rows"]) == prov["rows_canonical_sha256"]
    os.makedirs(f"{A}/{arm}", exist_ok=True)
    open(f"{A}/{arm}/cd16_protocol.json", "w").write(json.dumps(p, indent=2) + "\n")
    open(f"{A}/{arm}/cd16_serving.json", "w").write(json.dumps(s, indent=2) + "\n")
    print(arm, "ok")
