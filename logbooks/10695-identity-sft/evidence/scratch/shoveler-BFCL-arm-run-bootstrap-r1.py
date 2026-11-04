"""Path B BFCL run-cluster bootstrap for one of A13, M25, M49, M51, M75 (shoveler, after marten's A12 script b9566b6f).

Usage (from /home/karan/oa/marin, via the host-test-budget wrapper):
    python /tmp/genet-pathb-check.py /mnt/storage/karan-scratch/shoveler-BFCL-arm-run-bootstrap-r1.py ARM INDEX_PATH INDEX_SHA256 [--write]

Method, all frozen before outcomes:
- A13: d3c966d9 (primary minus fresh v0.1; secondary minus A8a only).
- Merges: 108c2f65 (primary minus fresh v0.1; secondary minus A3 and A8a; canonical order A12, A13, M25, M49, M51, M75).
- Reduced n: f44554ba (n0 all null; n1 mean/range only; n>=2 bootstrap the n actual admitted run fractions; the canonical
  stream always advances through the reserved fixed-n5 block, 10000 x 5 choice(range(5)) calls per arm).
The frozen prefix through A11 is replayed exactly by executing marten's hash-bound A11-minus-A9 script up to its `output`
assignment, as marten's A12 script does. Without --write it prints the result and writes nothing.
"""

import ast
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import pyarrow.ipc as ipc

ROOT = Path("/home/karan/Dropbox/marin-notes/snowball-identity/path-b")
HERE = ROOT / "bfcl-noise-r1"
ORDER = ["A12", "A13", "M25", "M49", "M51", "M75"]
A12_START = "0b389af49c68f2e9368db6e6419e5829365848133316ab47ea1b8ef03ff6eba3"
A13_START = "cde9ce3c079a3d40e708e9b4acffa5f71456393d563f7f15fa21fbc86c6ead02"
SPECS = {
    "A13": ("A12-A13-science-and-analysis-preregistration-marten-r1.json", "d3c966d90391dd7e1fabb5e6e29ecdb3382f454c71fd3a4a312878d03cea8738"),
    "merge": ("M25-M49-M51-M75-merge-sweep-preregistration-marten-r2.json", "108c2f65840ea7ab22acdb677494726a35d316e7400af5ba048337e84d44c4b9"),
}
ADDENDUM = ("BFCL-infra-exhaustion-analysis-addendum-marten-r1.json", "f44554ba02340cfcd8a787b0225fc7857e1558e084fb2c76d388b4a326cdfe91")
CONTRASTS = {"A12": ["fresh_v01", "A8a"], "A13": ["fresh_v01", "A8a"], "M25": ["fresh_v01", "A3", "A8a"], "M49": ["fresh_v01", "A3", "A8a"],
             "M51": ["fresh_v01", "A3", "A8a"], "M75": ["fresh_v01", "A3", "A8a"]}  # fmt: skip
PARENT = Path("/mnt/storage/karan-scratch/marten-bfcl-A11-minus-A9-final-r1.py")
JOINT = ("A11-minus-A9-registered-contrast-final-marten-r1.json", "4b22c9cd2cde36b57733ead5da59a9d10ec040c1ba89c8a72de895921b2cfc29")
ITERATIONS, RESERVED = 10000, 5


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def state_sha(rng):
    return digest(json.dumps(rng.getstate()).encode())


def checked(path, expected, length=None):
    raw = Path(path).read_bytes()
    assert digest(raw) == expected, str(path)
    assert length is None or len(raw) == length, str(path)
    return json.loads(raw)


def descriptor(entry):
    return checked(entry["path"], entry["sha256"], entry["bytes"])


def frozen_prefix():
    joint = checked(HERE / JOINT[0], JOINT[1])
    raw = PARENT.read_bytes()
    assert digest(raw) == joint["source_script_sha256"]
    module = ast.parse(raw)
    position = next(i for i, node in enumerate(module.body) if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "output" for t in node.targets))  # fmt: skip
    prior = {"__file__": str(PARENT), "__name__": "frozen_prefix_through_A11"}
    exec(compile(ast.Module(body=module.body[:position], type_ignores=[]), str(PARENT), "exec"), prior)
    assert prior["contrast"] == joint["contrast"]["A11_minus_A9"]
    assert state_sha(prior["canonical"]) == A12_START
    return prior


def advance_reserved_block(canonical):
    for _ in range(ITERATIONS):
        for _ in range(RESERVED):
            canonical.choice(range(RESERVED))


def qualify(index, arm, task_ids, baseline_model, baseline_eval):
    """Join every physical workflow's native record, launch, full JSON and typed projection; return per-job rows."""
    physical = {}
    pins = Counter()
    for entry in index["physical"]:
        proof_path = Path(entry["proof"])
        proof = checked(proof_path, entry["proof_sha256"])
        launch = checked(proof_path.with_name("launch.json"), proof["launch_sha256"])
        assert launch["job"] == launch["expected_job"] == proof["job"] == entry["job"]
        assert launch["application_submit_calls"] == 1
        pins[(launch["source_pin"], launch["local_native_workspace_bundle_sha256"], launch["HF"], launch["HF_readiness_proof_sha256"])] += 1
        assert launch["HF"] == proof["HF"] == proof["source_HF"]
        assert launch["HF_readiness_proof_sha256"] == proof["readiness_sha256"]
        assert proof["SDK_terminal"]
        assert proof["original_slot"] == entry["original_slot"] and proof["replacement_number"] == entry["replacement_number"]
        row = dict(slot=entry["original_slot"], replacement_number=entry["replacement_number"], job=entry["job"],
                   proof_sha256=entry["proof_sha256"], launch_sha256=proof["launch_sha256"], SDK_state=proof["SDK_state"])  # fmt: skip
        native_entry = proof.get("native_record_full_original_restricted_local")
        if native_entry is None:  # no native record at all: an unscored workflow, kept with explicit missingness
            assert not entry["primary_admissible"] and entry["score"] is None
            physical[entry["job"]] = dict(row, correct=None, scored=0, primary_admissible=False, native="absent")
            continue
        native = descriptor(native_entry)
        assert native["run_id"] == proof["native_eval_id"] == entry["native_eval_id"]
        assert native["results_path"] == proof["results_uri"] == entry["results_uri"]
        assert native["model"]["location"] == launch["HF"]
        assert native["model"]["config_digest"] == proof["native_model_config_sha256"]
        assert {k: v for k, v in native["model"]["config"].items() if k not in ["name", "location"]} == {
            k: v for k, v in baseline_model.items() if k not in ["name", "location"]}  # fmt: skip
        assert native["eval"] == baseline_eval
        samples = descriptor(proof["samples_full_JSON_native_reader_projection"])
        arrow = proof["samples_full_typed_native_reader_projection"]
        arrow_raw = Path(arrow["path"]).read_bytes()
        assert digest(arrow_raw) == arrow["sha256"] and len(arrow_raw) == arrow["bytes"]
        table = ipc.open_file(arrow["path"]).read_all()
        assert table.num_rows == 123 and len(table.column_names) == 23
        assert json.loads(json.dumps(table.to_pylist())) == samples
        assert len(samples) == len({s["doc_id"] for s in samples}) == 123 and {s["doc_id"] for s in samples} == task_ids
        rewards, errors, scored = {}, [], 0
        for sample in samples:
            detail = json.loads(sample["grading"]["detail"])
            values = [value for name, value in sample["metrics"] if name == "reward"]
            if detail["scored"]:
                assert len(values) == 1
                reward = values[0]
                assert type(reward) in (int, float) and math.isfinite(reward) and reward in (0, 1)
                assert detail["rewards"]["reward"] == sample["grading"]["score"] == reward
                scored += 1
            else:  # infrastructure-unscored: NULL, never zero
                assert not values and sample["grading"]["score"] is None and sample["grading"]["passed"] is None
                assert detail["error"]["category"] == "infrastructure"
                reward = None
            rewards[sample["doc_id"]] = reward
            if detail["error"] is not None:
                errors.append(dict(doc_id=sample["doc_id"], scored=detail["scored"], reward=reward,
                                   type=detail["error"]["type"], category=detail["error"]["category"]))  # fmt: skip
        assert {r["doc_id"]: r["reward"] for r in proof["rows"]} == rewards and len(proof["rows"]) == 123
        coverage = native["coverage"]["bfcl_parity"]
        assert coverage == proof["native_coverage"] and coverage["n_benchmark"] == 123
        assert scored == coverage["n_scored"] == proof["scored_binary_rewards"] == entry["scored_binary_rewards"]
        assert dict(Counter(e["type"] for e in errors)) == coverage["errors"]
        observed = int(sum(v for v in rewards.values() if v is not None))
        assert observed == proof["observed_correct"] == native["metrics"]["bfcl_parity"]["solved"]
        valid = scored == 123 and proof["SDK_state"] == "succeeded" and native["status"] == "succeeded"
        assert valid == proof["logical_full123_admissible"] == entry["primary_admissible"]
        assert (observed if valid else None) == proof["score"] == entry["score"]
        physical[entry["job"]] = dict(row, run_id=native["run_id"], correct=observed if valid else None, observed_correct=observed,
                                      scored=scored, native_record_sha256=proof["native_record_sha256"],
                                      JSON_sha256=proof["samples_full_JSON_native_reader_projection"]["sha256"],
                                      typed_reader_sha256=arrow["sha256"], native_scored_and_unscored_errors=errors,
                                      primary_admissible=valid)  # fmt: skip
    assert len(pins) == 1, f"{arm}: physical workflows disagree on source/bundle/HF/readiness pins: {pins}"
    return physical, next(iter(pins))


def logical_chains(index, physical):
    """One score per slot: the sole complete-123 run of the chain; a chain may end exhausted (no score), never past R2."""
    admitted, chains = [], []
    for entry in index["logical"]:
        chain = sorted([r for r in physical.values() if r["slot"] == entry["original_slot"]], key=lambda r: r["replacement_number"])
        assert [r["replacement_number"] for r in chain] == list(range(len(chain))) and len(chain) <= 3  # original, R1, R2
        valid = [r for r in chain if r["primary_admissible"]]
        assert len(valid) <= 1, f"slot {entry['original_slot']}: more than one valid score; disclosure required, no best-of"
        if valid:
            assert chain[-1] is valid[0], "a valid score was followed by a rerun"
            admitted.append(valid[0])
            status = "fulfilled"
        else:
            assert entry.get("status") == "exhausted", f"slot {entry['original_slot']} has no score and is not marked exhausted"
            status = "exhausted"
        chains.append(dict(slot=entry["original_slot"], status=status, jobs=[r["job"] for r in chain]))
    assert len(chains) == 5
    return admitted, chains


def interval(values):
    ordered = sorted(values)
    return [ordered[249], ordered[9749]]


def main():
    arm, index_path, index_sha = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    write = "--write" in sys.argv[4:]
    assert arm in ORDER
    spec_file, spec_sha = SPECS["A13" if arm in ("A12", "A13") else "merge"]
    checked(HERE / spec_file, spec_sha)
    checked(HERE / ADDENDUM[0], ADDENDUM[1])
    prior = frozen_prefix()
    canonical = prior["canonical"]
    starts = {}
    for name in ORDER:
        starts[name] = state_sha(canonical)
        if name == arm:
            break
        advance_reserved_block(canonical)
    assert starts["A12"] == A12_START and starts.get("A13", A13_START) == A13_START
    start_state = canonical.getstate()
    namespace = prior["namespace"]
    index = checked(index_path, index_sha)
    assert index["arm"] == arm and index["planned_logical_runs"] == 5 and len(index["logical"]) == 5
    assert index["all_physical_attempts_closed"] and index["missing_windows"] in (0, [])
    assert all(entry.get("status", "fulfilled") in ("fulfilled", "admitted", "exhausted") for entry in index["logical"]), "pending chains"
    physical, pins = qualify(index, arm, namespace["task_ids"], namespace["baseline_native_config"], namespace["baseline_eval_config"])
    admitted, chains = logical_chains(index, physical)
    admitted.sort(key=lambda r: r["run_id"])  # lexical actual admitted run IDs (f44554ba)
    job_order_equal = [r["job"] for r in admitted] == sorted(r["job"] for r in admitted)
    fractions = [r["correct"] / 123 for r in admitted]
    n = len(fractions)
    generator = random.Random()
    generator.setstate(start_state)
    draws = []
    for _ in range(ITERATIONS):
        picks = [generator.choice(fractions) for _ in range(n)] if n else []
        if n >= 2:
            draws.append(statistics.fmean(picks))
    advance_reserved_block(canonical)  # the canonical stream always consumes the reserved fixed-n5 block
    next_start = state_sha(canonical)
    if n == 5:
        assert generator.getstate() == canonical.getstate()
    mean = statistics.fmean(fractions) if n else None
    summary = dict(planned=5, n=n, exhausted_infra_no_score=sum(c["status"] == "exhausted" for c in chains), pending=0,
                   physical_workflows=len(physical), logical_chains=chains, runs=admitted,
                   mean_percent=None if mean is None else 100 * mean,
                   sample_SD_percent=100 * statistics.stdev(fractions) if n >= 2 else None,
                   min_max_percent=[100 * min(fractions), 100 * max(fractions)] if n else None,
                   unadjusted_mean_CI95_percent=[100 * v for v in interval(draws)] if n >= 2 else None,
                   null_reason=None if n >= 2 else ("no_complete123_run" if n == 0 else "fewer_than2_evaluation_clusters"))  # fmt: skip
    contrasts = {}
    for label in CONTRASTS[arm]:
        values, comparator_draws = prior["comparators"][label]
        point = None if mean is None else 100 * (mean - statistics.fmean(values))
        ci = [100 * v for v in interval([a - b for a, b in zip(draws, comparator_draws, strict=True)])] if n >= 2 else None
        contrasts[f"{arm}_minus_{label}"] = dict(delta_pp=point, unadjusted_CI95_pp=ci)
    for name, expected in prior["FROZEN"].items():
        assert digest((HERE / name).read_bytes()) == expected
    output = dict(
        status=f"FINAL_{arm}_ACTUAL_N{n}_REGISTERED_RUN_CLUSTER_BOOTSTRAP", arm=arm, analyst="shoveler",
        preregistration_sha256=spec_sha, reduced_n_addendum_sha256=ADDENDUM[1], index_path=str(index_path), index_sha256=index_sha,
        pins=dict(source_pin=pins[0], native_bundle_sha256=pins[1], HF=pins[2], HF_readiness_sha256=pins[3]),
        all_physical=list(physical.values()), summary=summary, contrasts=contrasts,
        primary=[f"{arm}_minus_fresh_v01"], secondary=[f"{arm}_minus_{c}" for c in CONTRASTS[arm][1:]],
        block_start_state_sha256=starts[arm], canonical_block_starts=starts, next_canonical_block_start_sha256=next_start,
        shared_draws_sha256=digest(json.dumps(draws).encode()) if n >= 2 else None,
        lexical_run_id_order_equals_job_order=job_order_equal,
        statistics="Frozen PythonRandom297 prefix through A11 replayed exactly; canonical stream advanced through the reserved n5 blocks in order A12, A13, M25, M49, M51, M75; this arm's analysis RNG cloned from its own block start; n choices per draw from lexically ordered actual admitted run fractions; shared draw j across registered contrasts with frozen comparator draw j; unadjusted sorted endpoints 249/9749; no item or repeat pairing.",
        model_network_storage_control_calls=0, new_retry_course_or_release_authority=False,
        source_script_sha256=digest(Path(__file__).read_bytes()),
    )  # fmt: skip
    text = json.dumps(output, indent=2) + "\n"
    print(json.dumps(dict(arm=arm, summary={k: v for k, v in summary.items() if k not in ("runs", "logical_chains")}, contrasts=contrasts,
                          start=starts[arm], next_start=next_start, job_order_equal=job_order_equal), indent=2))  # fmt: skip
    if arm == "A12":  # validation replay against marten's frozen A12 result; never written
        frozen = checked(HERE / "A12-run-bootstrap-extension-marten-r1.json", "1cf7f046cf6263e49fdca2d428571ed9b7245b671793f80da0d6afb0d890b225")
        assert frozen["summary"]["mean_percent"] == summary["mean_percent"]
        assert frozen["summary"]["unadjusted_mean_CI95_percent"] == summary["unadjusted_mean_CI95_percent"], "A12 CI does not replay"
        for key in contrasts:
            assert frozen["contrasts"][key] == {"delta_pp": contrasts[key]["delta_pp"], "unadjusted_CI95_pp": contrasts[key]["unadjusted_CI95_pp"]}, key
        assert frozen["A13_canonical_block_start_state_sha256"] == next_start == A13_START
        print("A12 REPLAY EXACT")
        return
    if write:
        destination = HERE / f"{arm}-run-bootstrap-extension-shoveler-r1.json"
        with destination.open("x") as stream:
            stream.write(text)
        print(json.dumps(dict(path=str(destination), sha256=digest(text.encode()))))


if __name__ == "__main__":
    main()
