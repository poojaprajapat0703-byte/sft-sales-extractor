"""evaluate.py - scores a predictions file against the held-out test set. No LLM judge needed.

predictions file: JSONL with {"id": ..., "output": "<raw model text>"}.
Metrics: json_valid, F1 (pain / signals / objections / stakeholder roles), decision accuracy + macro-F1,
grounding rate (quotes found verbatim), fabrication rate (gold-null budget/timeline that the model filled in).
Invalid outputs count as empty predictions (conservative).
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from schema import Extraction, parse_json, quote_in, collect_quotes, DECISIONS


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r)) if p + r else 0.0


def label_sets(x):
    return {"pain": {i["category"] for i in x["pain_points"]},
            "signal": {i["type"] for i in x["buying_signals"]},
            "objection": {i["category"] for i in x["objections"]},
            "role": {i["role"] for i in x["stakeholders"]}}


def evaluate(preds, test):
    n = len(test); valid = 0
    counts = {k: [0, 0, 0] for k in ("pain", "signal", "objection", "role")}  # tp, fp, fn
    dec_pairs, n_quotes, n_grounded, outputs_with_bad_quote = [], 0, 0, 0
    fab = nonnull_slots = missed = null_slots = 0
    failures = []
    for ex in test:
        gold = ex["gold"]; g = label_sets(gold)
        out = preds.get(ex["id"])
        pred, reasons = None, []
        try:
            pred = Extraction.model_validate(parse_json(out)).model_dump()
            valid += 1
        except Exception:
            reasons.append("invalid_json_or_schema")
        p = label_sets(pred) if pred else {k: set() for k in g}
        for k in counts:
            counts[k][0] += len(p[k] & g[k]); counts[k][1] += len(p[k] - g[k]); counts[k][2] += len(g[k] - p[k])
        dec_pairs.append((gold["decision"], pred["decision"] if pred else "invalid"))
        if pred:
            if pred["decision"] != gold["decision"]: reasons.append("wrong_decision")
            qs = collect_quotes(pred)
            bad = [q for q in qs if not quote_in(q, ex["transcript"])]
            n_quotes += len(qs); n_grounded += len(qs) - len(bad)
            if bad: outputs_with_bad_quote += 1; reasons.append("ungrounded_quote")
            for slot in ("budget", "timeline"):
                if gold[slot]["value"] is None:
                    null_slots += 1
                    if pred[slot]["value"] is not None: fab += 1; reasons.append(f"fabricated_{slot}")
                else:
                    nonnull_slots += 1
                    if pred[slot]["value"] is None: missed += 1; reasons.append(f"missed_{slot}")
        if reasons: failures.append({"id": ex["id"], "reasons": reasons})
    f1 = {k: round(prf(*v), 4) for k, v in counts.items()}
    acc = sum(a == b for a, b in dec_pairs) / n
    macro = sum(prf(sum(1 for a, b in dec_pairs if a == l and b == l),
                    sum(1 for a, b in dec_pairs if a != l and b == l),
                    sum(1 for a, b in dec_pairs if a == l and b != l)) for l in DECISIONS) / len(DECISIONS)
    return {"n": n, "json_valid_rate": round(valid / n, 4), "f1": f1,
            "decision_accuracy": round(acc, 4), "decision_macro_f1": round(macro, 4),
            "grounding_rate": round(n_grounded / n_quotes, 4) if n_quotes else None,
            "outputs_with_ungrounded_quote": round(outputs_with_bad_quote / max(valid, 1), 4),
            "fabrication_rate_on_null_slots": round(fab / null_slots, 4) if null_slots else None,
            "null_slots": null_slots, "missed_rate_on_stated_slots": round(missed / nonnull_slots, 4) if nonnull_slots else None,
            }, failures, dec_pairs


def bootstrap_ci(dec_pairs, iters=1000, seed=0):
    rng = random.Random(seed); n = len(dec_pairs); vals = []
    for _ in range(iters):
        s = [dec_pairs[rng.randrange(n)] for _ in range(n)]
        vals.append(sum(a == b for a, b in s) / n)
    vals.sort(); return [round(vals[int(.025 * iters)], 4), round(vals[int(.975 * iters)], 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--name", default="run")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    test = [json.loads(l) for l in open(a.test, encoding="utf8") if l.strip()]
    preds = {}
    for l in open(a.pred, encoding="utf8"):
        if l.strip():
            r = json.loads(l); preds[r["id"]] = r["output"]
    missing = [t["id"] for t in test if t["id"] not in preds]
    assert not missing, f"{len(missing)} test ids have no prediction (e.g. {missing[:3]})"
    metrics, failures, pairs = evaluate(preds, test)
    metrics["decision_accuracy_95ci"] = bootstrap_ci(pairs)
    metrics["name"] = a.name
    os.makedirs(a.outdir, exist_ok=True)
    json.dump(metrics, open(f"{a.outdir}/{a.name}.json", "w"), indent=2)
    with open(f"{a.outdir}/{a.name}_failures.jsonl", "w") as f:
        for x in failures: f.write(json.dumps(x) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
