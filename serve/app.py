"""serve/app.py - FastAPI endpoint: transcript in, validated + grounding-checked JSON out.

Run:  MODEL_PATH=outputs/gguf/model-q4_k_m.gguf uvicorn serve.app:app --port 8000
Test without a model:  MOCK_MODEL=1 uvicorn serve.app:app --port 8000   (returns a stub; for plumbing tests only)
Every request appends one JSON line to logs/requests.jsonl (basic request logging and failure tracking).
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fastapi import FastAPI
from pydantic import BaseModel
from schema import Extraction, make_messages, parse_json, collect_quotes, quote_in

MODEL_VERSION = os.environ.get("MODEL_VERSION", "qwen2.5-1.5b-sales-qlora-v1")
LOG_PATH = os.environ.get("LOG_PATH", "logs/requests.jsonl")
app = FastAPI(title="Sales-call extractor")
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        _llm = Llama(model_path=os.environ["MODEL_PATH"], n_ctx=4096, verbose=False)
    return _llm


def generate(transcript: str) -> str:
    if os.environ.get("MOCK_MODEL"):
        return json.dumps({"pain_points": [], "buying_signals": [], "objections": [], "stakeholders": [],
                           "budget": {"value": None, "evidence_quote": None},
                           "timeline": {"value": None, "evidence_quote": None}, "decision": "nurture"})
    r = get_llm().create_chat_completion(messages=make_messages(transcript), temperature=0.0, max_tokens=700)
    return r["choices"][0]["message"]["content"]


class Req(BaseModel):
    transcript: str


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/extract")
def extract(req: Req):
    t0 = time.time()
    raw, extraction, valid, ungrounded, error = "", None, False, [], None
    try:
        raw = generate(req.transcript)
        extraction = Extraction.model_validate(parse_json(raw)).model_dump()
        valid = True
        ungrounded = [q for q in collect_quotes(extraction) if not quote_in(q, req.transcript)]
    except Exception as e:  # invalid JSON, schema failure, model error
        error = f"{type(e).__name__}: {str(e)[:200]}"
    latency_ms = round((time.time() - t0) * 1000)
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf8") as f:
        f.write(json.dumps({"ts": time.time(), "model_version": MODEL_VERSION, "latency_ms": latency_ms,
                            "schema_valid": valid, "grounding_ok": valid and not ungrounded,
                            "n_ungrounded": len(ungrounded), "error": error, "transcript_chars": len(req.transcript)}) + "\n")
    return {"schema_valid": valid, "grounding_ok": valid and not ungrounded, "ungrounded_quotes": ungrounded,
            "extraction": extraction, "error": error, "latency_ms": latency_ms, "model_version": MODEL_VERSION}
