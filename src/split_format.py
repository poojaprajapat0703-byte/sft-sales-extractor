"""split_format.py - exact train/val/test split + chat-format JSONL + replay mix.

test = --test examples, val = --val examples, train = everything else (counts always sum to the clean total).
Outputs in data/: test.jsonl (id, transcript, gold), val.jsonl, train_no_replay.jsonl, train_mixed.jsonl (chat format),
split_report.json
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from schema import make_messages


def load_replay(k, seed):
    """k general-instruction examples from databricks-dolly-15k (CC BY-SA 3.0)."""
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = [r for r in ds if r["instruction"].strip() and not r["context"].strip() and 20 < len(r["response"]) < 1200]
    random.Random(seed).shuffle(rows)
    return [{"id": f"replay_{i}", "messages": [{"role": "user", "content": r["instruction"]},
                                              {"role": "assistant", "content": r["response"]}]}
            for i, r in enumerate(rows[:k])]


def write(path, rows):
    with open(path, "w", encoding="utf8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="data/clean.jsonl")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--test", type=int, default=80)
    ap.add_argument("--val", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--replay-frac", type=float, default=0.10, help="replay examples as fraction of train size")
    ap.add_argument("--no-replay", action="store_true", help="skip downloading dolly (e.g. offline test)")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.clean, encoding="utf8") if l.strip()]
    random.Random(a.seed).shuffle(rows)
    assert len(rows) > a.test + a.val + 10, f"only {len(rows)} clean examples; need more data or smaller --test/--val"
    test, val, train = rows[:a.test], rows[a.test:a.test + a.val], rows[a.test + a.val:]
    ids = [set(r["id"] for r in s) for s in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2]), "split overlap!"
    assert len(train) + len(val) + len(test) == len(rows)

    to_chat = lambda r: {"id": r["id"], "messages": make_messages(r["transcript"], r["gold"])}
    train_chat, val_chat = [to_chat(r) for r in train], [to_chat(r) for r in val]
    os.makedirs(a.outdir, exist_ok=True)
    write(f"{a.outdir}/test.jsonl", test)  # keeps gold separately for evaluation
    write(f"{a.outdir}/val.jsonl", val_chat)
    write(f"{a.outdir}/train_no_replay.jsonl", train_chat)
    n_replay = 0
    if not a.no_replay:
        replay = load_replay(round(a.replay_frac * len(train)), a.seed)
        n_replay = len(replay)
        mixed = train_chat + replay
        random.Random(a.seed).shuffle(mixed)
        write(f"{a.outdir}/train_mixed.jsonl", mixed)
    rep = {"clean_total": len(rows), "train": len(train), "val": len(val), "test": len(test), "replay_added": n_replay}
    json.dump(rep, open(f"{a.outdir}/split_report.json", "w"), indent=2)
    print(rep)


if __name__ == "__main__":
    main()
