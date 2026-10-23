# Grounded B2B sales-call extractor (QLoRA fine-tune)

Fine-tunes `Qwen2.5-1.5B-Instruct` (Apache-2.0) with QLoRA/SFT to turn a B2B discovery-call transcript into strict JSON
(pain points, buying signals, objections, stakeholders, budget, timeline, qualify/nurture/disqualify), where every claim
carries a verbatim evidence quote and missing facts must be `null`.

**Data is synthetic** (spec-driven, written by an open-weight teacher model). Labels come from the generation spec, quotes
are machine-verified. It does not capture real call messiness. Say this first in any interview.

## What I have and have not tested
The CPU pieces (schema, filtering, split, evaluation, API plumbing) were tested end-to-end on mock data.
**The GPU scripts (train.py, infer.py, merge_export.py) were NOT run on a GPU by the author of this code.**
Unsloth / TRL / transformers change quickly; expect to fix an argument or version error. Run the smoke test first.

## Order of work
Phases 1-3 run on your laptop (CPU). Phases 4-8 run on a Kaggle GPU notebook (T4, internet on).

1. **Setup**: `pip install -r requirements-data.txt`; copy `.env.example` to `.env` and export `GROQ_API_KEY`. Check your Groq
   console for an open-weight model and read its license; set `GROQ_MODEL`.
2. **Test the pipeline with no API** (mock data, never train on it):
   `python src/generate_data.py --mock --n 150 --out data/raw_mock.jsonl && python src/filter_data.py --raw data/raw_mock.jsonl --out data/clean_mock.jsonl --tfidf --dup-threshold 0.97`
3. **Generate real data** (resumable, re-run after rate limits): `python src/generate_data.py --n 700`
4. **Filter**: `python src/filter_data.py` (MiniLM dedupe; first run downloads a small model). Keep `data/filter_stats.json` for the README funnel.
5. **Split + format**: `python src/split_format.py --test 80 --val 30` (train = the rest; replay = 10% Dolly). Hand-check 50 test examples.
6. **Upload `data/` and `src/` to Kaggle** (Datasets, private). In the notebook: `!pip install unsloth lm-eval datasets sentence-transformers`
7. **Baselines**:
   ```
   python src/infer.py --mode zero --out outputs/pred_zero.jsonl && python src/evaluate.py --pred outputs/pred_zero.jsonl --name zero
   python src/infer.py --mode few  --out outputs/pred_few.jsonl  && python src/evaluate.py --pred outputs/pred_few.jsonl  --name few
   ```
8. **Smoke test, then train**:
   ```
   python src/train.py --smoke
   python src/train.py --train data/train_mixed.jsonl     --out outputs/adapter_replay
   python src/train.py --train data/train_no_replay.jsonl --out outputs/adapter_noreplay   # ablation
   ```
   Read the `MASK CHECK` line: prompt tokens must be masked. Download the adapter folders before the session ends.
9. **Evaluate tuned**:
   ```
   python src/infer.py --mode tuned --adapter outputs/adapter_replay --out outputs/pred_tuned_replay.jsonl
   python src/evaluate.py --pred outputs/pred_tuned_replay.jsonl --name tuned_replay      # repeat for noreplay
   python src/compare.py
   ```
   Read `results/*_failures.jsonl`, bucket 20 failures by reason, write MODEL_REPORT.md.
10. **Regression check**: `bash scripts/run_regression.sh` (small subsets; indicative only).
11. **Deploy (stretch)**: `python src/merge_export.py --adapter outputs/adapter_replay`, then
    `MODEL_PATH=outputs/gguf/<file>.gguf uvicorn serve.app:app --port 8000`, test with
    `curl -X POST localhost:8000/extract -H "content-type: application/json" -d '{"transcript":"Rep: ..."}'`.
    Docker: see the Dockerfile header. `python serve/summarize_logs.py` summarizes request logs.

## Labeling rule
poor fit -> `disqualify`; good fit AND budget stated AND >=2 buying signals -> `qualify`; otherwise `nurture`.

## Metrics (src/evaluate.py)
JSON/schema validity; F1 for pain/signal/objection/stakeholder-role sets; decision accuracy and macro-F1 (+ bootstrap CI);
grounding rate (quotes found verbatim in transcript); fabrication rate (budget/timeline filled in when the gold is null).

## Limitations (fill in with real numbers)
Synthetic data; small test set (differences under ~5 points are noise); regression subsets are small; quantization effect
must be measured on your GGUF before claiming anything about it.
