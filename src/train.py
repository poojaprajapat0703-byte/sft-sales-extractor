"""train.py - QLoRA supervised fine-tuning (SFT) with Unsloth + TRL.

  python src/train.py --smoke                                   # 20 examples, 20 steps: ALWAYS run this first
  python src/train.py --train data/train_mixed.jsonl --out outputs/adapter_replay
  python src/train.py --train data/train_no_replay.jsonl --out outputs/adapter_noreplay   # ablation

What each piece does:
  4-bit base model  -> "Q" in QLoRA: the frozen base is stored in 4 bits so it fits a 16 GB T4
  LoRA r/alpha      -> small trainable low-rank matrices added to every linear layer
  train_on_responses_only -> loss masking: learn only the assistant answer, not the system/user prompt
"""
import argparse, json, os, sys
from unsloth import FastLanguageModel, is_bfloat16_supported  # import unsloth FIRST
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


def build(cls, **kw):
    """TRL/transformers rename arguments between versions; try the known variants."""
    renames = [{}, {"max_seq_length": "max_length"}, {"eval_strategy": "evaluation_strategy"},
               {"max_seq_length": "max_length", "eval_strategy": "evaluation_strategy"}]
    err = None
    for r in renames:
        try:
            return cls(**{r.get(k, k): v for k, v in kw.items()})
        except TypeError as e:
            err = e
    raise err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train", default="data/train_mixed.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="outputs/adapter")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--push", default=None, help="HF repo id, e.g. yourname/sales-extractor-lora (needs HF_TOKEN)")
    a = ap.parse_args()

    model, tok = FastLanguageModel.from_pretrained(model_name=a.model, max_seq_length=a.max_seq,
                                                   load_in_4bit=True, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=a.r, lora_alpha=a.alpha, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)

    ds = load_dataset("json", data_files={"train": a.train, "validation": a.val})
    to_text = lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}
    ds = ds.map(to_text, remove_columns=ds["train"].column_names)
    if a.smoke:
        ds["train"] = ds["train"].select(range(min(20, len(ds["train"]))))
        ds["validation"] = ds["validation"].select(range(min(4, len(ds["validation"]))))

    bf16 = is_bfloat16_supported()  # False on T4 -> fp16
    cfg = build(SFTConfig, output_dir=a.out + "_ckpt", dataset_text_field="text", max_seq_length=a.max_seq,
                per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum,
                num_train_epochs=a.epochs, max_steps=20 if a.smoke else -1, learning_rate=a.lr,
                warmup_ratio=0.05, lr_scheduler_type="linear", weight_decay=0.01, optim="adamw_8bit",
                fp16=not bf16, bf16=bf16, logging_steps=5, eval_strategy="steps", eval_steps=25,
                per_device_eval_batch_size=1, save_strategy="no", seed=3407, report_to="none",
                packing=False, dataset_num_proc=2)
    try:
        trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds["train"], eval_dataset=ds["validation"], args=cfg)
    except TypeError:  # newer TRL: tokenizer -> processing_class
        trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds["train"], eval_dataset=ds["validation"], args=cfg)

    # LOSS MASKING: only the assistant answer contributes to the loss (Qwen/ChatML markers)
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                      response_part="<|im_start|>assistant\n")
    ex = trainer.train_dataset[0]
    labels = [t for t in ex["labels"] if t != -100]
    masked = sum(1 for t in ex["labels"] if t == -100)
    print(f"\nMASK CHECK: {masked} prompt tokens masked, {len(labels)} answer tokens trained.")
    print("Trained-on text starts with:", repr(tok.decode(labels[:40])))
    assert masked > 0 and len(labels) > 0, "loss masking failed - do not continue"

    trainer.train()
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    json.dump(trainer.state.log_history, open(os.path.join(a.out, "log_history.json"), "w"), indent=1)
    try:  # loss curve for your README
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        h = trainer.state.log_history
        plt.plot([x["step"] for x in h if "loss" in x], [x["loss"] for x in h if "loss" in x], label="train")
        plt.plot([x["step"] for x in h if "eval_loss" in x], [x["eval_loss"] for x in h if "eval_loss" in x], label="val")
        plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.savefig(os.path.join(a.out, "loss_curve.png"), dpi=120)
    except Exception as e:
        print("plot skipped:", e)
    if a.push:
        model.push_to_hub(a.push, token=os.environ.get("HF_TOKEN")); tok.push_to_hub(a.push, token=os.environ.get("HF_TOKEN"))
    print("Saved adapter to", a.out)


if __name__ == "__main__":
    main()
