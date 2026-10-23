"""filter_data.py - quality filtering with counts after every stage.

Stages: build gold -> schema valid -> quotes grounded -> length -> absent-fact leak -> near-duplicates.
Writes data/clean.jsonl and data/filter_stats.json (put the funnel in your README).
"""
import argparse, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from schema import Extraction, build_gold, quote_in, collect_quotes


def embed(texts, use_tfidf):
    if use_tfidf:  # lightweight fallback, no model download
        from sklearn.feature_extraction.text import TfidfVectorizer
        m = TfidfVectorizer(max_features=20000, ngram_range=(1, 2)).fit_transform(texts)
        m = m.toarray().astype("float32")
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw.jsonl")
    ap.add_argument("--out", default="data/clean.jsonl")
    ap.add_argument("--stats", default="data/filter_stats.json")
    ap.add_argument("--dup-threshold", type=float, default=0.90)
    ap.add_argument("--min-words", type=int, default=350)
    ap.add_argument("--max-words", type=int, default=1300)
    ap.add_argument("--tfidf", action="store_true", help="use TF-IDF instead of MiniLM for dedupe")
    a = ap.parse_args()

    raw = [json.loads(l) for l in open(a.raw, encoding="utf8") if l.strip()]
    stats = {"raw": len(raw)}
    dropped = {}
    keep = []

    def drop(reason): dropped[reason] = dropped.get(reason, 0) + 1

    for r in raw:
        try:
            gold = build_gold(r["spec"], r["evidence"])          # missing evidence key -> KeyError
        except (KeyError, TypeError):
            drop("missing_evidence"); continue
        try:
            Extraction.model_validate(gold)
        except Exception:
            drop("schema_invalid"); continue
        if not all(quote_in(q, r["transcript"]) for q in collect_quotes(gold)):
            drop("quote_not_in_transcript"); continue
        n_words = len(r["transcript"].split())
        if not (a.min_words <= n_words <= a.max_words):
            drop("length"); continue
        if gold["budget"]["value"] is None and re.search(r"\$\s?\d", r["transcript"]):
            drop("budget_leak_in_null_case"); continue          # gold says no budget but call shows one
        keep.append({"id": r["id"], "transcript": r["transcript"], "gold": gold})
    stats["after_rule_filters"] = len(keep)

    # near-duplicate removal (greedy: keep the first of any similar pair)
    if keep:
        X = embed([k["transcript"] for k in keep], a.tfidf)
        kept_idx = []
        for i in range(len(keep)):
            if kept_idx and float(np.max(X[kept_idx] @ X[i])) >= a.dup_threshold:
                drop("near_duplicate"); continue
            kept_idx.append(i)
        keep = [keep[i] for i in kept_idx]
    stats["after_dedupe"] = len(keep)
    stats["dropped_by_reason"] = dropped

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf8") as f:
        for k in keep: f.write(json.dumps(k, ensure_ascii=False) + "\n")
    json.dump(stats, open(a.stats, "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
