"""Summarize logs/requests.jsonl: volume, failure rates, latency percentiles per model version."""
import json, sys, statistics as st
path = sys.argv[1] if len(sys.argv) > 1 else "logs/requests.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]
if not rows: sys.exit("no requests logged yet")
by = {}
for r in rows: by.setdefault(r["model_version"], []).append(r)
for v, rs in by.items():
    lat = sorted(r["latency_ms"] for r in rs)
    pct = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]
    print(f"model {v}: {len(rs)} requests")
    print(f"  schema failures : {sum(not r['schema_valid'] for r in rs) / len(rs):.1%}")
    print(f"  grounding fails : {sum(not r['grounding_ok'] for r in rs) / len(rs):.1%}")
    print(f"  latency ms      : p50={pct(.5)} p95={pct(.95)} mean={st.mean(lat):.0f}")
