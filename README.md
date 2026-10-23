
# Grounded B2B Sales-Call Extractor (QLoRA Fine-Tuning)

SalesSignal AI fine-tunes `Qwen2.5-1.5B-Instruct` using QLoRA/SFT to turn B2B discovery-call transcripts into structured JSON: pain points, buying signals, objections, stakeholders, budget, timeline, and a qualify/nurture/disqualify decision.

Every claim should carry a verbatim evidence quote. Missing facts must be `null`.

## Data and Testing Disclaimer

**The dataset is synthetic.** It is generated using a spec-driven process and an open-weight teacher model. It does not represent the full messiness of real sales calls. Be transparent about this when presenting the project.

The original project documentation states that CPU components were tested on mock data, but the GPU scripts (`train.py`, `infer.py`, `merge_export.py`) were not verified on a GPU. Run the smoke test before full training.

## Workflow

1. Install dependencies: `pip install -r requirements-data.txt`.
2. Copy `.env.example` to `.env` and configure `GROQ_API_KEY` and `GROQ_MODEL`.
3. Test the pipeline using mock data. Do not use mock data for training.
4. Generate synthetic training data with `python src/generate_data.py --n 700`.
5. Filter data with `python src/filter_data.py`.
6. Split and format data with `python src/split_format.py --test 80 --val 30`.
7. Upload the required data and source files to a private Kaggle dataset.
8. On Kaggle, install the training dependencies and run the baseline inference.
9. Run `python src/train.py --smoke` before full training.
10. Train, evaluate, and compare the tuned model against the baselines.
11. Run regression checks and consider deployment after validating the GPU workflow.

Refer to the individual scripts for their exact command-line arguments and outputs.

## Decision Rule

- Poor fit → `disqualify`
- Good fit, budget stated, and at least two buying signals → `qualify`
- Otherwise → `nurture`

## Evaluation Metrics

- JSON and schema validity
- F1 scores for pain points, buying signals, objections, and stakeholder roles
- Decision accuracy and macro-F1
- Grounding rate: whether evidence quotes appear verbatim in the transcript
- Fabrication rate: whether budget or timeline is filled in when the gold label is `null`

## Limitations

- Synthetic data may not represent real-world call complexity.
- A small test set produces noisy comparisons.
- Regression subsets may be too small for strong conclusions.
- Quantization effects must be measured before making performance claims.
- GPU training and export compatibility must be verified in the actual environment.
