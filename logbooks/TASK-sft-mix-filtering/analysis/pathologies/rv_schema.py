import sys, pyarrow.parquet as pq
t = pq.read_table(sys.argv[1]); print(t.num_rows); print(t.schema)
r = t.slice(0,1).to_pylist()[0]; print({k: str(v)[:300] for k, v in r.items()})
