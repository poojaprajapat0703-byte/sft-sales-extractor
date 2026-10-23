"""schema.py - the single source of truth for the whole project.

Used by: data generation, filtering, evaluation and the API.
Closed label lists (PAIN, SIGNALS, ...) make scoring objective: we can compute F1 without an LLM judge.
"""
import json, re
from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict

PAIN = ["manual_reporting", "slow_onboarding", "high_churn", "data_silos",
        "compliance_risk", "rising_costs", "poor_forecasting", "tool_sprawl"]
SIGNALS = ["budget_confirmed", "timeline_urgency", "champion_identified",
           "exec_sponsor", "competitor_mentioned", "pilot_requested"]
OBJECTIONS = ["price", "timing", "security_review", "integration_effort",
              "incumbent_vendor", "no_champion"]
ROLES = ["cfo", "cto", "vp_sales", "ops_manager", "it_security", "procurement", "founder"]
STANCES = ["champion", "neutral", "skeptic"]
DECISIONS = ["qualify", "nurture", "disqualify"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown keys = invalid output


class PainPoint(Strict):
    category: Literal[tuple(PAIN)]
    evidence_quote: str


class Signal(Strict):
    type: Literal[tuple(SIGNALS)]
    evidence_quote: str


class Objection(Strict):
    category: Literal[tuple(OBJECTIONS)]
    evidence_quote: str


class Stakeholder(Strict):
    role: Literal[tuple(ROLES)]
    stance: Literal[tuple(STANCES)]


class Slot(Strict):
    """budget / timeline. BOTH fields must be null when the call never states it."""
    value: Optional[str] = None
    evidence_quote: Optional[str] = None


class Extraction(Strict):
    pain_points: List[PainPoint]
    buying_signals: List[Signal]
    objections: List[Objection]
    stakeholders: List[Stakeholder]
    budget: Slot
    timeline: Slot
    decision: Literal[tuple(DECISIONS)]


SYSTEM_PROMPT = (
    "You extract structured sales intelligence from a B2B discovery-call transcript. "
    "Return ONLY one valid JSON object, no extra text. "
    "Every evidence_quote must be copied exactly (verbatim) from the transcript. "
    "If budget or timeline is not stated in the call, set both its value and evidence_quote to null. "
    f"pain_points.category in {PAIN}. buying_signals.type in {SIGNALS}. "
    f"objections.category in {OBJECTIONS}. stakeholders.role in {ROLES}, stance in {STANCES}. "
    f"decision in {DECISIONS}. "
    "JSON keys: pain_points, buying_signals, objections, stakeholders, budget, timeline, decision."
)


def make_messages(transcript: str, gold: Optional[dict] = None):
    """Chat format used for training AND inference (they must match exactly)."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Transcript:\n" + transcript}]
    if gold is not None:
        msgs.append({"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)})
    return msgs


# ---------- labeling rule (this is your written "labeling rule") ----------
def decide(fit: str, budget_stated: bool, n_signals: int) -> str:
    """poor fit -> disqualify; good fit + budget + >=2 signals -> qualify; otherwise nurture."""
    if fit == "poor":
        return "disqualify"
    if fit == "good" and budget_stated and n_signals >= 2:
        return "qualify"
    return "nurture"


def build_gold(spec: dict, evidence: dict) -> dict:
    """Gold labels come from the SPEC (known by construction), quotes from the teacher.
    Raises KeyError if the teacher did not return a quote for a fact."""
    return {
        "pain_points": [{"category": c, "evidence_quote": evidence[f"pain:{c}"]} for c in spec["pains"]],
        "buying_signals": [{"type": s, "evidence_quote": evidence[f"signal:{s}"]} for s in spec["signals"]],
        "objections": [{"category": o, "evidence_quote": evidence[f"objection:{o}"]} for o in spec["objections"]],
        "stakeholders": [{"role": r, "stance": st} for r, st in spec["stakeholders"]],
        "budget": ({"value": spec["budget"], "evidence_quote": evidence["budget"]}
                   if spec["budget"] else {"value": None, "evidence_quote": None}),
        "timeline": ({"value": spec["timeline"], "evidence_quote": evidence["timeline"]}
                     if spec["timeline"] else {"value": None, "evidence_quote": None}),
        "decision": spec["decision"],
    }


# ---------- grounding helpers ----------
def normalize(s: str) -> str:
    s = s.lower().replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", s).strip()


def quote_in(quote: str, transcript: str) -> bool:
    return bool(quote) and normalize(quote) in normalize(transcript)


def collect_quotes(extraction: dict) -> List[str]:
    q = []
    for key in ("pain_points", "buying_signals", "objections"):
        q += [x["evidence_quote"] for x in extraction.get(key, [])]
    for key in ("budget", "timeline"):
        if extraction.get(key, {}).get("evidence_quote"):
            q.append(extraction[key]["evidence_quote"])
    return q


def parse_json(text: str):
    """Pull the first JSON object out of model text (handles ``` fences and <think> blocks)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])
