"""generate_data.py - spec-driven synthetic data generation.

1) sample a random SPEC (which facts are present/absent, true decision)
2) ask an open-weight TEACHER model (via Groq free tier) to write a call transcript containing exactly those facts
   and to return the exact sentence for each fact
3) append to data/raw.jsonl after EVERY example (safe to stop and resume)

--mock builds fake template transcripts with NO API call. Use it ONLY to test the pipeline, never to train.
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from schema import PAIN, SIGNALS, OBJECTIONS, ROLES, decide, parse_json

VERTICALS = ["fintech", "healthcare SaaS", "manufacturing", "logistics", "retail e-commerce",
             "edtech", "insurance", "telecom", "legal tech", "construction"]
SIZES = ["10-50 employees", "50-200 employees", "200-1000 employees", "1000+ employees"]
BUDGETS = ["$20k-30k per year", "$50k-80k per year", "around $120k annually", "a $15k pilot budget"]
TIMELINES = ["go live by the end of next quarter", "a decision within six weeks",
             "start a pilot next month", "implement before the fiscal year ends in March"]

HINTS = {
    "pain": {"manual_reporting": "analysts spend days building reports by hand",
             "slow_onboarding": "new customers take too long to onboard",
             "high_churn": "customers are leaving faster than expected",
             "data_silos": "teams cannot see each other's data",
             "compliance_risk": "audits and compliance are a worry",
             "rising_costs": "operating costs keep climbing",
             "poor_forecasting": "forecasts are routinely wrong",
             "tool_sprawl": "too many overlapping tools"},
    "signal": {"budget_confirmed": "prospect confirms budget is approved",
               "timeline_urgency": "prospect has a firm deadline",
               "champion_identified": "a prospect contact says they will push this internally",
               "exec_sponsor": "an executive is backing the project",
               "competitor_mentioned": "prospect says they are also evaluating a named competitor",
               "pilot_requested": "prospect asks to run a pilot"},
    "objection": {"price": "worried it is too expensive", "timing": "not the right time",
                  "security_review": "needs a security review first",
                  "integration_effort": "worried about integration effort",
                  "incumbent_vendor": "already has a vendor they like",
                  "no_champion": "nobody internally wants to own it"},
}


def sample_spec(rng: random.Random) -> dict:
    fit = rng.choices(["good", "medium", "poor"], [0.40, 0.35, 0.25])[0]
    n_sig = {"good": rng.randint(2, 4), "medium": rng.randint(1, 2), "poor": rng.randint(0, 1)}[fit]
    signals = rng.sample(SIGNALS, n_sig)
    p_stated = {"good": 0.65, "medium": 0.40, "poor": 0.20}[fit]
    budget = rng.choice(BUDGETS) if ("budget_confirmed" in signals or rng.random() < p_stated) else None
    timeline = rng.choice(TIMELINES) if ("timeline_urgency" in signals or rng.random() < p_stated) else None
    if not budget and "budget_confirmed" in signals: signals.remove("budget_confirmed")
    if not timeline and "timeline_urgency" in signals: signals.remove("timeline_urgency")
    n_obj = {"good": rng.randint(0, 1), "medium": rng.randint(0, 2), "poor": rng.randint(1, 2)}[fit]
    objections = rng.sample(OBJECTIONS, n_obj)
    if "champion_identified" in signals and "no_champion" in objections: objections.remove("no_champion")
    roles = rng.sample(ROLES, rng.randint(1, 3))
    if "exec_sponsor" in signals and not any(r in roles for r in ("cfo", "cto", "founder")):
        roles[0] = rng.choice(["cfo", "cto", "founder"])
    stakeholders = [[r, rng.choice(["champion", "neutral", "skeptic"])] for r in roles]
    if "no_champion" in objections:
        stakeholders = [[r, "neutral" if s == "champion" else s] for r, s in stakeholders]
    if "champion_identified" in signals:
        stakeholders[0][1] = "champion"
    return {"vertical": rng.choice(VERTICALS), "size": rng.choice(SIZES), "fit": fit,
            "pains": rng.sample(PAIN, rng.randint(1, 3)), "signals": signals, "objections": objections,
            "stakeholders": stakeholders, "budget": budget, "timeline": timeline,
            "decision": decide(fit, bool(budget), len(signals))}


def teacher_prompt(spec: dict) -> str:
    facts = {}
    for c in spec["pains"]: facts[f"pain:{c}"] = f"Prospect describes a problem: {HINTS['pain'][c]}"
    for s in spec["signals"]: facts[f"signal:{s}"] = f"Buying signal: {HINTS['signal'][s]}"
    for o in spec["objections"]: facts[f"objection:{o}"] = f"Objection: {HINTS['objection'][o]}"
    if spec["budget"]: facts["budget"] = f"Prospect states a budget: {spec['budget']}"
    if spec["timeline"]: facts["timeline"] = f"Prospect states a timeline: {spec['timeline']}"
    who = "; ".join(f"{r} ({s})" for r, s in spec["stakeholders"])
    absent = []
    if not spec["budget"]: absent.append("any budget, price range or dollar amount")
    if not spec["timeline"]: absent.append("any deadline or timeline")
    return f"""Write a realistic B2B sales DISCOVERY CALL transcript between a Rep and the Prospect(s).
Company: {spec['vertical']}, {spec['size']}. People on the prospect side: {who}. Make their attitudes show naturally.
Length: 600 to 900 words. Format every line as 'Rep: ...' or 'Prospect (<role>): ...'. Vary names and style.
The call MUST clearly contain these facts, each expressed in ONE natural sentence a person would say:
{json.dumps(facts, indent=1)}
The call must NOT mention: {', '.join(absent) if absent else 'nothing extra'}. Do not add other buying signals or objections beyond those listed.
Return ONLY JSON: {{"transcript": "<full transcript with \\n between lines>", "evidence": {{"<fact key>": "<the exact sentence copied verbatim from your transcript>"}}}}
Include one evidence entry for EVERY fact key above."""


def call_teacher(client, model, spec):
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": teacher_prompt(spec)}],
        temperature=0.8, max_tokens=2200, response_format={"type": "json_object"})
    out = parse_json(r.choices[0].message.content)
    return out["transcript"], out["evidence"]


def mock_teacher(spec):
    """Fake transcript for pipeline tests only."""
    lines, ev = ["Rep: Thanks for making time today, tell me about your setup."], {}
    def add(key, sentence):
        lines.append(f"Prospect: {sentence}"); ev[key] = sentence
    for c in spec["pains"]: add(f"pain:{c}", f"Honestly the biggest issue is that {HINTS['pain'][c]} at our {spec['vertical']} company.")
    for s in spec["signals"]: add(f"signal:{s}", f"For us it matters that {HINTS['signal'][s]}, case {random.random():.4f}.")
    for o in spec["objections"]: add(f"objection:{o}", f"One concern is that we are {HINTS['objection'][o]}, noted {random.random():.4f}.")
    if spec["budget"]: add("budget", f"We have set aside {spec['budget']} for this.")
    if spec["timeline"]: add("timeline", f"We would need to {spec['timeline']}.")
    lines.append(" ".join(f"Rep: Understood, tell me more about point {i} {random.random():.4f}." for i in range(70)))
    return "\n".join(lines), ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--out", default="data/raw.jsonl")
    ap.add_argument("--model", default=os.environ.get("GROQ_MODEL", "qwen/qwen3-32b"),
                    help="open-weight teacher listed in YOUR Groq console; check its license")
    ap.add_argument("--sleep", type=float, default=2.5, help="seconds between calls (free-tier rate limit)")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out, encoding="utf8") if l.strip()}
    client = None
    if not a.mock:
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
    fails = 0
    for i in range(a.n):
        eid = f"ex_{i:05d}"
        if eid in done: continue
        spec = sample_spec(random.Random(1000 + i))  # deterministic per id
        try:
            tr, ev = mock_teacher(spec) if a.mock else call_teacher(client, a.model, spec)
            fails = 0
        except Exception as e:  # rate limit, bad JSON, network ...
            fails += 1
            print(f"[{eid}] failed ({type(e).__name__}): {str(e)[:120]}")
            if fails >= 8:
                print("Too many failures in a row (likely free-tier limit). Re-run the same command later; it resumes."); break
            time.sleep(min(60, 5 * 2 ** fails)); continue
        with open(a.out, "a", encoding="utf8") as f:
            f.write(json.dumps({"id": eid, "spec": spec, "transcript": tr, "evidence": ev,
                                "teacher": "mock" if a.mock else a.model}, ensure_ascii=False) + "\n")
        print(f"[{eid}] ok")
        if not a.mock: time.sleep(a.sleep)


if __name__ == "__main__":
    main()
