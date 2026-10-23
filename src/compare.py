"""compare.py - print a markdown results table from results/*.json (paste into README / MODEL_REPORT.md)."""
import glob, json, sys
names = sys.argv[1:] or ["zero", "few", "tuned_replay", "tuned_noreplay"]
rows = []
for n in names:
    try: rows.append(json.load(open(f"results/{n}.json")))
    except FileNotFoundError: print(f"(missing results/{n}.json)", file=sys.stderr)
cols = [("JSON valid", lambda m: m["json_valid_rate"]), ("Pain F1", lambda m: m["f1"]["pain"]),
        ("Signal F1", lambda m: m["f1"]["signal"]), ("Objection F1", lambda m: m["f1"]["objection"]),
        ("Decision acc", lambda m: m["decision_accuracy"]), ("Decision macro-F1", lambda m: m["decision_macro_f1"]),
        ("Grounding", lambda m: m["grounding_rate"]), ("Fabrication (lower better)", lambda m: m["fabrication_rate_on_null_slots"])]
print("| System | " + " | ".join(c for c, _ in cols) + " |\n|" + "---|" * (len(cols) + 1))
for m in rows: print(f"| {m['name']} | " + " | ".join(str(f(m)) for _, f in cols) + " |")
print(f"\nn = {rows[0]['n']} test examples; differences under ~5 points are likely noise.") if rows else None
