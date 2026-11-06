#!/usr/bin/env python3
"""Hero free-memory gate from DCGM. No code in the job.

usage: gate_dcgm.py <train job path> [--at YYYY-MM-DDTHH:MM:SS] [--minutes N]
Run from a marin checkout. Prints the job's GPU count, max used and min free over the window.
Nodes come from the latest attempt's bootstrap log lines plus a live or saved IP-to-node map.
"""
import argparse, datetime as dt, json, os, re, subprocess, sys

GIB = 1024**3
TOTAL_GIB = 184.30  # free + pool_reserved + non_pool on all 704 devices in arm 3
EXPECTED_GPUS = 704
HERE = os.path.dirname(os.path.abspath(__file__))


def query(sql):
    r = subprocess.run(["uv", "run", "finelog", "query", "marin", sql], capture_output=True, text=True, timeout=240)
    if r.returncode:
        sys.exit(f"finelog failed: {r.stderr[-400:]}")
    return [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]


def job_nodes(job):
    ip2node = json.load(open(os.path.join(HERE, "ip2node.json")))
    rows = query(f"""SELECT key, data FROM "log" WHERE key LIKE '{job}/%' AND data LIKE '%rank0%initialize_jax bootstrap inputs%' LIMIT 20000""")
    latest = {}
    for row in rows:
        task, attempt = row["key"].rsplit(":", 1)
        m = re.search(r"advertise_host=([\d.]+)", row["data"])
        if m:
            a = int(attempt)
            if task not in latest or a >= latest[task][0]:
                latest[task] = (a, m[1])
    missing = sorted({ip for _, ip in latest.values() if ip not in ip2node})
    if missing:
        sys.exit(f"{len(missing)} of {len(latest)} task IPs are not in ip2node.json, refresh the map: {missing[:3]}")
    return sorted({ip2node[ip] for _, ip in latest.values()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("job")
    p.add_argument("--at", help="window end, UTC; default now")
    p.add_argument("--minutes", type=int, default=5)
    a = p.parse_args()
    end = dt.datetime.fromisoformat(a.at).replace(tzinfo=dt.UTC) if a.at else dt.datetime.now(dt.UTC)
    e = int(end.timestamp() * 1000)
    s = e - a.minutes * 60000
    nodes = job_nodes(a.job)
    names = ",".join(f"'{n}'" for n in nodes)
    rows = query(f"""SELECT node_name, json_get(attributes_json,'gpu_index') AS gpu, max(value) AS used, count(*) AS n
      FROM "telemetry_v1.node_agent"
      WHERE name='gpu_memory_used_bytes' AND timestamp_ms >= {s} AND timestamp_ms < {e} AND node_name IN ({names})
      GROUP BY 1,2""")
    used = sorted(r["used"] / GIB for r in rows)
    if not used:
        sys.exit("no DCGM rows in the window")
    print(f"window end {end:%H:%M:%S}Z, {a.minutes} min; nodes {len(nodes)}, gpus {len(used)} of {EXPECTED_GPUS}")
    print(f"max used {used[-1]:.3f} GiB; min free {TOTAL_GIB - used[-1]:.3f} GiB; median used {used[len(used)//2]:.3f} GiB")
    if len(used) != EXPECTED_GPUS:
        print("INVALID: wrong GPU count, read again")


main()
