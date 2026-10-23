"""infer.py - run the base model (zero-shot / few-shot) or the fine-tuned adapter on the test set. GPU needed (Kaggle T4).

Same 4-bit loading, same greedy decoding (do_sample=False) for ALL systems = fair comparison.
  python src/infer.py --mode zero  --out outputs/pred_zero.jsonl
  python src/infer.py --mode few   --out outputs/pred_few.jsonl
  python src/infer.py --mode tuned --adapter outputs/adapter_replay --out outputs/pred_tuned.jsonl
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from unsloth import FastLanguageModel  # import unsloth FIRST
import torch
from schema import make_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zero", "few", "tuned"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--train", default="data/train_no_replay.jsonl", help="source of few-shot examples")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=700)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.mode == "tuned": assert a.adapter, "--adapter is required for tuned mode"

    test = [json.loads(l) for l in open(a.test, encoding="utf8") if l.strip()][:a.limit]
    shots = []
    if a.mode == "few":  # 2 fixed examples from TRAIN (never from test)
        shots = [json.loads(l)["messages"] for l in open(a.train, encoding="utf8")][:2]

    model, tok = FastLanguageModel.from_pretrained(
        model_name=a.adapter if a.mode == "tuned" else a.model,
        max_seq_length=a.max_seq, load_in_4bit=True, dtype=None)
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    def prompt(ex):
        msgs = make_messages(ex["transcript"])  # system + user
        if shots:  # insert (user, assistant) pairs after the system message
            fs = []
            for s in shots: fs += s[1:3]
            msgs = [msgs[0]] + fs + [msgs[1]]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf8") as f:
        for i in range(0, len(test), a.bs):
            batch = test[i:i + a.bs]
            enc = tok([prompt(e) for e in batch], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, use_cache=True,
                                     pad_token_id=tok.pad_token_id)
            for e, g in zip(batch, gen):
                text = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                f.write(json.dumps({"id": e["id"], "output": text}, ensure_ascii=False) + "\n")
            print(f"{min(i + a.bs, len(test))}/{len(test)}", flush=True)


if __name__ == "__main__":
    main()
