"""usage: compose_a13.py PACKET_DIR. Compose the A13 training-root request at 69ddc88f offline (placeholder credentials, no submit).

Re-verifies the sealed review packet with the old worktree prefix remapped to the 69ddc88f checkout,
then calls ablation_launch.submission with the plan's A13 training kwargs exactly as prepare_request did.
"""
import copy, hashlib, json, sys
from pathlib import Path

PACKET = Path(sys.argv[1])
OLD, NEW = "/home/karan/oa/marin-path-b-a12-a13/", "/mnt/storage/karan-scratchpad/task-334/a13-compose/src/"
from experiments.post_training.identity_path_b.ablation_launch import submission

receipt = json.loads((PACKET / "review-receipt.json").read_text())
ok = bad = 0
for path, item in receipt["files"].items():
    p = Path(path.replace(OLD, NEW))
    raw = p.read_bytes()
    good = len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"]
    ok += good; bad += not good
    if not good: print("MISMATCH", path)
print(f"packet files verified {ok}, mismatched {bad}")
plan = json.loads((PACKET / "launch-plan.json").read_text())
target = plan["arm"] if "arm" in plan and isinstance(plan["arm"], dict) else next(a for a in plan["arms"] if a["arm"] == "A13")
training = copy.deepcopy(target["training_kwargs"])
training["runtime_bytes"] = Path(plan["runtime"]["local"]).read_bytes()
req = submission(name=target["root_name"], training=training, credentials={"CW_KEY_ID": "placeholder", "CW_KEY_SECRET": "placeholder"})
print("name", req["name"])
print("retries failure/preemption", req["max_retries_failure"], req["max_retries_preemption"])
print("priority", req["priority_band"], "timeout", req["timeout"])
print("resources", str(req["resources"]).replace("\n", " "))
print("env vars", sorted(req["environment"].env_vars))
print("secret names", sorted(req["environment"].secrets))
print("run_root", training["run_root"])
print("packed", training["packed_sha256"], "pin", training["training_pin"])
print("COMPOSE_OK" if bad == 0 else "COMPOSE_PACKET_MISMATCH")
