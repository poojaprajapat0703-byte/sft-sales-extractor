"""merge_export.py - merge LoRA into the base model and export a quantized GGUF for CPU serving."""
import argparse
from unsloth import FastLanguageModel

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--merged", default="outputs/merged")
ap.add_argument("--gguf", default="outputs/gguf")
ap.add_argument("--quant", default="q4_k_m")
a = ap.parse_args()
model, tok = FastLanguageModel.from_pretrained(model_name=a.adapter, max_seq_length=4096, load_in_4bit=True)
model.save_pretrained_merged(a.merged, tok, save_method="merged_16bit")   # full merged weights
model.save_pretrained_gguf(a.gguf, tok, quantization_method=a.quant)       # GGUF for llama.cpp
print("done:", a.merged, a.gguf)
