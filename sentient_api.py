"""Sentient AI specialist-matching service for the Milestone 3 n8n workflow.

Wraps the frozen Milestone 2 model (MLP 64, run mlp_64_v1) and the deterministic
safety gate from Sentient_AI_Milestone_2.ipynb behind two HTTP endpoints:

    POST /safety-gate   deterministic emergency / review screening (no model, no LLM)
    POST /match         pre-decision truncation + MLP 64 specialty prediction

Run from the root of the Zoticus-AI repo:
    pip install -r requirements.txt
    uvicorn sentient_api:app --host 127.0.0.1 --port 8000

The service receives de-identified clinical text only. It never receives or stores
patient names or email addresses, and it keeps no request logs.
"""
import os
import re
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

_RELATIVE_MODEL = Path("experiments") / "milestone2_resub_v13_seed101" / "frozen_model.joblib"


def _find_model():
    """Use SENTIENT_MODEL_PATH if set, else look next to this script, then in the current folder."""
    if os.environ.get("SENTIENT_MODEL_PATH"):
        candidates = [Path(os.environ["SENTIENT_MODEL_PATH"])]
    else:
        candidates = [Path(__file__).resolve().parent / _RELATIVE_MODEL, Path.cwd() / _RELATIVE_MODEL]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "frozen_model.joblib not found. Looked in:\n  "
        + "\n  ".join(str(p) for p in candidates)
        + "\nPut sentient_api.py in the root of your Zoticus-AI repo, or set "
          "SENTIENT_MODEL_PATH to the full path of frozen_model.joblib."
    )


MODEL_PATH = _find_model()
MODEL_RUN_ID = "mlp_64_v1"
MIN_PREDECISION_CHARS = 40

TARGET_CLASSES = [
    "Cardiovascular / Pulmonary", "Orthopedic", "Gastroenterology", "Neurology",
    "Obstetrics / Gynecology", "Urology", "ENT - Otolaryngology", "Nephrology",
    "Dermatology", "Ophthalmology",
]

_MODEL = joblib.load(MODEL_PATH)
LABELS = [str(c) for c in _MODEL.classes_]
if sorted(LABELS) != sorted(TARGET_CLASSES):
    raise RuntimeError(f"Model classes do not match the supported specialties: {LABELS}")

# ---------------------------------------------------------------------------
# Safety gate: ported unchanged from Sentient_AI_Milestone_2.ipynb
# ---------------------------------------------------------------------------
RED_FLAGS = re.compile(
    r"\b(?:"
    r"chest (?:pain|pressure)|"
    r"(?:acute|severe) shortness of breath|"
    r"difficulty breathing|(?:cannot|can't) breathe|"
    r"sudden weakness|one[- ]sided weakness|"
    r"sudden speech difficulty|"
    r"uncontrolled bleeding|loss of consciousness"
    r")\b",
    re.IGNORECASE,
)

EMERGENCY_GUIDANCE = (
    "Your reported symptoms may require emergency care. "
    "Call 911 immediately in the United States, or your local "
    "emergency number. Do not wait for a specialist appointment."
)

VITAL_PATTERNS = {
    "BP": r"\b(?:BP|blood pressure)\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})",
    "HR": r"\b(?:HR|heart rate|pulse)\s*[:=]?\s*(\d{2,3})",
    "SpO2": r"\b(?:SpO2|oxygen saturation)\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)\s*%?",
    "RR": r"\b(?:RR|respiratory rate)\s*[:=]?\s*(\d{1,3})",
    "Temp": r"\b(?:Temp(?:erature)?)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*°?\s*([CF])?\b",
}


def _base_safety_gate(text, *, screening_complete=False):
    if not isinstance(text, str) or not text.strip():
        return "REVIEW_REQUIRED"
    uncertain = False
    clauses = re.split(r"[.!?;,\n]|\b(?:but|however)\b", text.lower())
    for clause in clauses:
        for match in RED_FLAGS.finditer(clause):
            prefix = clause[:match.start()]
            negated = re.search(r"\b(?:no|denies|without)\s*$", prefix)
            unclear = re.search(r"\b(?:history of|previously|resolved|maybe|unsure)\b", clause)
            if negated or unclear:
                uncertain = True
                continue
            return "EMERGENCY_ESCALATED"
    if uncertain or not screening_complete:
        return "REVIEW_REQUIRED"
    return "NONEMERGENCY_ELIGIBLE"


def parse_vitals(text):
    readings = []
    for clause in re.split(r"[;\n]|(?<=[.!?])\s+", text):
        for kind, regex in VITAL_PATTERNS.items():
            for m in re.finditer(regex, clause, re.I):
                readings.append({
                    "kind": kind,
                    "values": m.groups(),
                    "uncertain_context": bool(re.search(
                        r"\b(history|previously|resolved|denies|not|no|family)\b", clause, re.I)),
                })
    return readings


def safety_gate(text, *, screening_complete=False):
    status = _base_safety_gate(text, screening_complete=screening_complete)
    if status == "EMERGENCY_ESCALATED" or not isinstance(text, str):
        return status
    readings = parse_vitals(text)
    for r in readings:
        if r["uncertain_context"]:
            continue
        value = float(r["values"][0])
        if ((r["kind"] == "BP" and 0 < value < 100) or
                (r["kind"] == "SpO2" and 0 < value < 90) or
                (r["kind"] == "RR" and value > 22)):
            return "EMERGENCY_ESCALATED"
    # HR/temperature thresholds remain reviewer-dependent.
    return "REVIEW_REQUIRED" if readings else status


# ---------------------------------------------------------------------------
# Pre-decision truncation: ported unchanged from the notebook
# ---------------------------------------------------------------------------
DECISION_HEADERS = [
    "PREOPERATIVE DIAGNOSIS", "PREOPERATIVE DIAGNOSES",
    "POSTOPERATIVE DIAGNOSIS", "POSTOPERATIVE DIAGNOSES",
    "PROCEDURE PERFORMED", "PROCEDURE IN DETAIL",
    "DESCRIPTION OF PROCEDURE", "PROCEDURE", "PROCEDURES",
    "ANESTHESIA", "FINDINGS", "IMPRESSION",
    "ASSESSMENT AND PLAN", "ASSESSMENT",
    "DIAGNOSIS", "DIAGNOSES", "PLAN",
    "COMPLICATIONS", "ESTIMATED BLOOD LOSS", "SPECIMENS",
    "INDICATIONS", "INDICATION", "DISPOSITION",
    "RECOMMENDATION", "RECOMMENDATIONS",
    "DISCHARGE DIAGNOSIS", "DISCHARGE DIAGNOSES",
    "FINAL DIAGNOSIS", "FINAL DIAGNOSES",
    "HOSPITAL COURSE", "DELIVERY NOTE",
]
_HEADER_PATTERN = re.compile(
    r"(?:^|(?<=[.,;:!?\n\r]))\s*(?P<header>"
    + "|".join(r"\s+".join(re.escape(w) for w in h.split())
               for h in sorted(set(DECISION_HEADERS), key=len, reverse=True))
    + r")\s*[:,]",
    re.IGNORECASE,
)


def truncate_pre_decision(text):
    if not isinstance(text, str) or not text.strip():
        return "", False, None
    match = _HEADER_PATTERN.search(text)
    if match:
        return text[:match.start()].strip(), True, " ".join(match.group("header").upper().split())
    return text.strip(), False, None


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(title="Sentient AI specialist matching", version="1.0")


class SafetyRequest(BaseModel):
    case_id: str
    text: str
    screening_complete: bool = False


class MatchRequest(BaseModel):
    case_id: str
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "model_run_id": MODEL_RUN_ID, "labels": LABELS}


@app.post("/safety-gate")
def run_safety_gate(req: SafetyRequest):
    status = safety_gate(req.text, screening_complete=req.screening_complete)
    guidance = {
        "EMERGENCY_ESCALATED": EMERGENCY_GUIDANCE,
        "REVIEW_REQUIRED": "Screening requires review. Specialty matching and booking are blocked.",
        "NONEMERGENCY_ELIGIBLE": None,
    }[status]
    return {"case_id": req.case_id, "safety_status": status, "guidance": guidance}


@app.post("/match")
def match_specialty(req: MatchRequest):
    kept, truncated, cut_header = truncate_pre_decision(req.text)
    base = {"case_id": req.case_id, "model_run_id": MODEL_RUN_ID,
            "truncated": truncated, "cut_header": cut_header}
    if len(kept) < MIN_PREDECISION_CHARS:
        return {**base, "match_status": "REVIEW_REQUIRED", "specialty": None,
                "confidence": None, "top_candidates": [],
                "reason": f"Pre-decision text shorter than {MIN_PREDECISION_CHARS} characters"}
    probs = _MODEL.predict_proba([kept])[0]
    ranked = sorted(zip(LABELS, map(float, probs)), key=lambda p: p[1], reverse=True)
    return {**base, "match_status": "MATCHED", "specialty": ranked[0][0],
            "confidence": round(ranked[0][1], 4),
            "top_candidates": [{"specialty": s, "probability": round(p, 4)} for s, p in ranked[:3]],
            "reason": None}