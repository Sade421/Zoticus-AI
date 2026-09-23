from __future__ import annotations

import re
from typing import Dict, List


SYMPTOM_OPTIONS: Dict[str, Dict[str, List[str]]] = {
    "head": {
        "pain": [
            "sharp pain in one side of the head",
            "pressure behind the eyes",
            "new severe headache",
            "head trauma or injury",
        ],
        "ache": [
            "dull headache",
            "pressure behind the eyes",
            "tension-like ache",
            "sinus pressure",
        ],
        "soreness": [
            "sore scalp",
            "jaw soreness",
            "neck and head soreness",
            "tenderness after stress or strain",
        ],
    },
    "shoulders": {
        "pain": [
            "sharp pain when lifting the arm",
            "pain radiating from the shoulder blade",
            "rotator cuff pain",
            "stiff joint pain",
        ],
        "ache": [
            "dull shoulder ache after activity",
            "muscle tightness",
            "general soreness across the shoulder",
            "aching with movement",
        ],
        "soreness": [
            "sore muscles after exercise",
            "tender soreness at the top of the shoulder",
            "painful range of motion",
            "soreness after repetitive work",
        ],
    },
    "back": {
        "pain": [
            "sharp pain traveling down the leg",
            "severe lower back pain",
            "muscle strain",
            "pain after lifting",
        ],
        "ache": [
            "dull ache in the lower back",
            "stiffness and tightness",
            "aching after sitting a long time",
            "general back tension",
        ],
        "soreness": [
            "sore muscles after activity",
            "tender spot in the middle back",
            "back soreness after poor posture",
            "soreness with twisting",
        ],
    },
    "stomach": {
        "pain": [
            "cramping after eating",
            "severe stomach pain",
            "burning pain in upper abdomen",
            "sharp pain near the lower belly",
        ],
        "ache": [
            "dull abdominal ache",
            "pressure in the upper belly",
            "bloating with discomfort",
            "aching after meals",
        ],
        "soreness": [
            "sore abdominal muscles",
            "tenderness with movement",
            "pressure around the stomach",
            "soreness after exercise",
        ],
        "heart": [
            "chest pain",
            "palpitations",
            "shortness of breath",
            "dizziness",
        ],
        "Skin": [
            "rash",
            "itching",  
            "redness",
            "blisters",
        ],
        "Hormones": [
            "irregular periods",
            "fatigue",
            "weight changes",
            "mood swings",
        ],
        "Cancer": [
            "unexplained weight loss",
            "persistent fatigue",
            "lumps or swelling",
            "changes in skin or moles",
        ],
        "Children": [
            "fever",
            "cough",
            "ear pain",
            "vomiting",
        ],
        "Pregnancy": [
            "morning sickness",
            "fatigue",
            "back pain",
            "swelling",
        ],
        "Mental Health": [
            "persistent sadness",
            "anxiety",
            "mood swings",
            "difficulty concentrating",
        ],

    },
}


SPECIALIST_BY_CONDITION: Dict[str, str] = {
    "head": "Neurologist",
    "shoulders": "Orthopedic Specialist",
    "back": "Orthopedic Spine Specialist",
    "stomach": "Gastroenterologist",
    "heart": "Cardiologist",
    "Skin": "Dermatologist",
    "Hormones": "Endocrinologist",
    "Cancer": "Oncologist",
    "Children": "Pediatrician",
    "Pregnancy": "Obstetrician/Gynecologist",
    "Mental Health": "Psychiatrist",

}


def get_symptom_options(body_part: str, feeling_type: str) -> List[str]:
    normalized_body = (body_part or "").strip().lower()
    normalized_feeling = (feeling_type or "").strip().lower()
    options = SYMPTOM_OPTIONS.get(normalized_body, {}).get(normalized_feeling, ["general discomfort"])
    return list(dict.fromkeys(options))


def parse_test_results(report_text: str) -> Dict[str, object]:
    """Extract explicitly written report findings without interpreting them."""
    text = (report_text or "").strip()
    findings: List[Dict[str, str]] = []
    current_modality = "Other"
    lab_test_pattern = re.compile(
        r"^(wbc|rbc|hemoglobin|hgb|platelets?|glucose|sodium|potassium|creatinine|troponin)\b",
        re.IGNORECASE,
    )
    modality_pattern = re.compile(
        r"\b(ct|mri|x-ray|xray|ultrasound|ecg|ekg|lab|laboratory|blood work)\b",
        re.IGNORECASE,
    )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        modality_match = modality_pattern.search(line)
        if modality_match:
            current_modality = modality_match.group(1).upper()

        if ":" in line:
            test_name, finding = (part.strip() for part in line.split(":", 1))
        else:
            test_name, finding = current_modality, line

        if lab_test_pattern.match(test_name):
            current_modality = "LAB"

        if finding:
            findings.append(
                {
                    "modality": current_modality,
                    "test": test_name or current_modality,
                    "finding": finding,
                    "source_text": line,
                }
            )

    return {"source_text": text, "findings": findings}


def summarize_diagnosis(document_text: str) -> str:
    parsed = parse_test_results(document_text)
    findings = parsed["findings"]
    if not findings:
        return "No structured test findings were provided. A clinician should review any available report text alongside the symptoms."

    lines = ["Structured test findings (not a diagnosis):"]
    for finding in findings:
        lines.append(f"- {finding['modality']} | {finding['test']}: {finding['finding']}")
    return "\n".join(lines)


def build_appointment_plan(
    body_part: str,
    feeling_type: str,
    symptom_detail: str,
    extra_info: str,
    test_results: str,
) -> Dict[str, object]:
    normalized_body = (body_part or "").strip().lower()
    normalized_feeling = (feeling_type or "").strip().lower()
    specialist = SPECIALIST_BY_CONDITION.get(normalized_body, "Primary Care Physician")

    symptom_summary = (symptom_detail or "general discomfort").strip() or "general discomfort"
    additional_info = (extra_info or "The patient is seeking specialist review for ongoing symptoms.").strip()
    tests = (test_results or "No specific test results provided.").strip()
    structured_tests = parse_test_results(test_results)
    structured_test_text = summarize_diagnosis(test_results)

    if normalized_body == "stomach":
        diagnosis_summary = (
            f"The stomach {normalized_feeling} described as '{symptom_summary}' should be reviewed with the patient's history and any gastrointestinal symptoms. "
            f"Additional context: {additional_info}. Test review: {tests}. {structured_test_text}"
        )
        before_appointment_instructions = [
            "Drink plenty of fluids in the day leading up to the appointment.",
            "Fast for 8 to 12 hours if your clinician specifically instructs it before bloodwork or abdominal imaging.",
            "Avoid strenuous exercise the day before your visit.",
            "Bring a list of all medications, supplements, and prior test results.",
        ]
        driving_instructions = [
            "You may be able to drive if you feel well and no sedation or procedure is planned.",
            "If an imaging study or procedure requires sedation, arrange a ride home instead of driving.",
        ]
        after_care_instructions = [
            "Rest and continue hydration after the visit.",
            "Follow up if abdominal pain worsens, you develop vomiting, fever, or black stools.",
            "Avoid heavy exercise until your clinician clears it.",
        ]
    elif normalized_body == "head":
        diagnosis_summary = (
            f"This headache or head {normalized_feeling} described as '{symptom_summary}' should be assessed for severity, duration, and red flags. "
            f"Additional context: {additional_info}. Test review: {tests}. {structured_test_text}"
        )
        before_appointment_instructions = [
            "Drink plenty of water and avoid missing meals.",
            "Avoid heavy exercise, alcohol, or dehydration before the appointment.",
            "Bring any imaging reports, medication list, and a symptom timeline.",
        ]
        driving_instructions = [
            "Do not drive if you feel faint, dizzy, or have had a recent procedure or sedating medication.",
            "When in doubt, arrange a ride for safety.",
        ]
        after_care_instructions = [
            "Rest and monitor for worsening headache, weakness, confusion, or vomiting.",
            "Follow up promptly if symptoms escalate or if new neurologic symptoms appear.",
            "Limit intense activity until the specialist gives guidance.",
        ]
    elif normalized_body == "back":
        diagnosis_summary = (
            f"The back {normalized_feeling} described as '{symptom_summary}' may reflect a musculoskeletal or nerve-related issue. "
            f"Additional context: {additional_info}. Test review: {tests}. {structured_test_text}"
        )
        before_appointment_instructions = [
            "Drink plenty of fluids and keep moving gently if it is safe.",
            "Avoid lifting heavy objects and stop exercise the day before the appointment.",
            "Bring your test results, medication list, and any notes about the pain pattern.",
        ]
        driving_instructions = [
            "You can usually drive if you are not in severe pain or taking medication that causes drowsiness.",
            "Use a ride if pain, weakness, or dizziness makes driving unsafe.",
        ]
        after_care_instructions = [
            "Use heat or ice as directed and avoid heavy lifting during recovery.",
            "Follow up if numbness, weakness, or bladder issues develop.",
            "Resume exercise gradually and only when symptoms improve.",
        ]

    else:
        diagnosis_summary = (
            f"The shoulder {normalized_feeling} described as '{symptom_summary}' should be assessed in the context of motion, injury, or daily activity. "
            f"Additional context: {additional_info}. Test review: {tests}. {structured_test_text}"
        )
        before_appointment_instructions = [
            "Stay hydrated and avoid repeated overhead lifting before the visit.",
            "Limit exercise the day before if the shoulder symptoms are active.",
            "Bring the test results, injury history, and current treatment plan.",
        ]
        driving_instructions = [
            "You may drive unless you are taking pain medication that causes drowsiness or you are in severe pain.",
            "Arrange a ride if the pain or medication makes it unsafe.",
        ]
        after_care_instructions = [
            "Rest the shoulder and avoid lifting beyond light activity until the clinician advises otherwise.",
            "Follow up if swelling, weakness, or persistent pain continues.",
            "Reintroduce exercise gradually and stop if symptoms flare.",
        ]

    return {
        "body_part": body_part.title(),
        "feeling_type": feeling_type.title(),
        "symptom_detail": symptom_summary,
        "diagnosis_summary": diagnosis_summary,
        "specialist": specialist,
        "recommended_visit": "Specialist consultation",
        "before_appointment_instructions": before_appointment_instructions,
        "driving_instructions": driving_instructions,
        "after_care_instructions": after_care_instructions,
        "test_results": tests,
        "structured_test_findings": structured_tests,
        "additional_info": additional_info,
    }


__all__ = [
    "get_symptom_options",
    "parse_test_results",
    "summarize_diagnosis",
    "build_appointment_plan",
    "SYMPTOM_OPTIONS",
]

