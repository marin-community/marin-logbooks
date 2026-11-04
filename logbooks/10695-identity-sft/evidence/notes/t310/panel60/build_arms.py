"""Build the panel-60 per-arm protocol and serving files from Ben's gist copies.

Usage: python3 build_arms.py BEN_GIST_DIR OUT_DIR. Only served_model/run (protocol) and name/location/revision (serving)
change; the 60 identity_rows are copied unchanged and their canonical sha256 (sort_keys, compact, ASCII) is asserted
to equal Ben's provenance rows_canonical_sha256 0fafe46f...
"""
import copy, hashlib, json, os, sys

G, A = sys.argv[1], sys.argv[2]
proto = json.load(open(f"{G}/protocol.json"))
serve = json.load(open(f"{G}/serving.json"))
prov = json.load(open(f"{G}/provenance.json"))
canon = lambda rows: hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
assert canon(proto["identity_rows"]) == prov["rows_canonical_sha256"]
ARMS = {
    "v0p1": ("snowball-v0p1-panel60", "open-athena/Marin-Snowball-67B-A2B-v0.1", "adc0078a19a4383bc998fcdb43867b3360baf5f3"),
    "v0p1-sysdefault": ("snowball-v0p1-sysdefault-panel60",
                        "s3://marin-rno2a/marin/checkpoints/snowball-identity-sysdefault/v0.1-sysdefault/hf", None),
    "opd20": ("snowball-opd20-panel60",
              "s3://marin-rno2a/marin/users/karan/checkpoints/snowball-identity-opd-cd16/train/2026.10.26.5/exports/global_step_20/policy", None),
    "cd16t2": ("snowball-cd16t2-panel60", "s3://marin-rno2a/marin/experiments/snowball-identity/t312/cd16-t2/train-r1/hf/step-15", None),
    "cd16t3": ("snowball-cd16t3-panel60", "s3://marin-rno2a/marin/experiments/snowball-identity/t312/cd16-t3/train-r1/hf/step-15", None),
    "held": ("grug-datakit-sft-262k-20261005-panel60", "s3://marin-rno2a/marin/models/grug-datakit-sft-262k-20261005-ac60d940-r1", None),
    "pathb-base10-u140": ("snowball-pathb-base10-u140-panel60",
                          "s3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/b-base-resume-r4-4015908e50e3/arm10/run/hf/updates-140", None),
    "pathb-base25-u205": ("snowball-pathb-base25-u205-panel60",
                          "s3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/b-base-resume-r4-4015908e50e3/arm25/run/hf/updates-205", None),
    "pathb-base25-u52": ("snowball-pathb-base25-u52-panel60",
                         "s3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/recovery-r1-d836c814f566/arm25-updates52/hf/updates-52", None),
}
for arm, (name, location, revision) in ARMS.items():
    p = copy.deepcopy(proto); p["served_model"] = name; p["run"] = f"panel60-{arm}-20261026"
    s = copy.deepcopy(serve); s["model"].update(name=name, location=location, revision=revision)
    assert canon(p["identity_rows"]) == prov["rows_canonical_sha256"]
    os.makedirs(f"{A}/{arm}", exist_ok=True)
    open(f"{A}/{arm}/cd16_protocol.json", "w").write(json.dumps(p, indent=2) + "\n")
    open(f"{A}/{arm}/cd16_serving.json", "w").write(json.dumps(s, indent=2) + "\n")
    print(arm, "ok")
