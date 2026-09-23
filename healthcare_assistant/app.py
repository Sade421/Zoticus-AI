import os
import re
from pathlib import Path
from typing import Optional

import streamlit as st

from symptom import build_appointment_plan, get_symptom_options, summarize_diagnosis
from specialty_classifier import build_case_text, predict_specialty


URGENT_PATTERNS = [
    "chest pain",
    "trouble breathing",
    "severe bleeding",
    "fainting",
    "confusion",
    "severe headache",
    "stroke",
    "unconscious",
    "suicidal",
    "high fever",
    "difficulty breathing",
    "shortness of breath",
]


def get_missing_information_question(
    symptom: str,
    duration: str,
    severity: Optional[int],
    location: str,
) -> Optional[str]:
    """Return the next single follow-up question needed for specialty review."""
    if not (symptom or "").strip():
        return "What is the main symptom you are experiencing?"
    if not (duration or "").strip():
        return "How long have you had this symptom?"
    if severity is None or severity <= 0:
        return "How severe is the symptom from 1-10?"
    if not (location or "").strip():
        return "Where is the symptom located?"
    return None


def generate_openai_response(prompt: str) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical information assistant. Provide general, cautious health information. "
                        "Do not diagnose, do not provide treatment directives beyond general, non-urgent guidance. "
                        "Always encourage contacting a qualified clinician for serious symptoms."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return getattr(response, "output_text", None) or response.choices[0].message.content
    except Exception:
        return None







st.set_page_config(page_title=" Sentient AI Healthcare Assistant", page_icon="🩺")
st.title("Sentient AI Healthcare Assistant")
st.caption("General clinical support for questions, document summaries, specialist scheduling, and clinician handoffs.")
st.warning("This tool is for informational use only and does not replace professional medical judgment.")

intake_tab, main_tab, summary_tab, specialist_tab, handoff_tab = st.tabs(
    
    [
        "Patient intake form",
        "Medical question",
        "Clinical document summary",
        "Specialist appointment planning",
        "Clinician handoff note",
    ]
)
with intake_tab:
    st.subheader("Patient intake form")
    st.caption("Enter the patient's current information for clinician review.")

    with st.form("patient_intake_form"):
        patient_name = st.text_input("Patient name (optional)")
        age = st.number_input("Age", min_value=0, max_value=150, step=1, value=0)
        sex = st.selectbox(
            "Sex",
            ["Prefer not to say", "Female", "Male", "Intersex", "Other"],
        )
        symptom_location = st.selectbox(
            "Symptom location",
            ["Head", "Chest", "Abdomen", "Back", "Shoulders", "Extremities", "Other"],
        )
        symptom_type = st.selectbox(
            "Symptom type",
            ["Pain", "Ache", "Pressure", "Burning", "Tingling", "Soreness", "Other"],
        )
        severity = st.slider("Severity", min_value=0, max_value=10, value=0, help="0 = no discomfort, 10 = worst possible")
        duration = st.text_input("Duration", placeholder="For example: 3 days or since this morning")
        main_complaint = st.text_area("Main complaint", height=100)
        medical_history = st.text_area("Medical history", height=120)
        medications = st.text_area("Medications", height=120)
        clinical_documents = st.file_uploader(
            "Upload clinical documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Accepted formats: PDF, DOCX, and TXT.",
        )
        test_results = st.text_area("Test-result text", height=140)
        submitted = st.form_submit_button("Submit intake")

    if submitted:
        uploaded_names = [document.name for document in clinical_documents]
        st.success("Patient intake submitted for clinician review.")
        st.write(
            {
                "Patient name": patient_name or "Not provided",
                "Age": age,
                "Sex": sex,
                "Main complaint": main_complaint or "Not provided",
                "Symptom location": symptom_location,
                "Symptom type": symptom_type,
                "Severity": f"{severity}/10",
                "Duration": duration or "Not provided",
                "Medical history": medical_history or "Not provided",
                "Medications": medications or "Not provided",
                "Uploaded clinical documents": uploaded_names or ["None"],
                "Test-result text": test_results or "Not provided",
            }
        )

with main_tab:
    st.subheader("Medical question")
    question = st.text_area("Enter a question about symptoms, care, or common health concerns:", height=150)
    if st.button("Get answer"):
        result = answer_medical_question(question)
        st.markdown(result)

with summary_tab:
    st.subheader("Clinical document summary")
    uploaded_document = st.file_uploader(
        "Upload a clinical note",
        type=["pdf", "docx", "txt"],
        help="Accepted formats: PDF, DOCX, and TXT.",
    )
    raw_text = st.text_area("Paste patient notes, visit summaries, or a chart excerpt:", height=220)
    if st.button("Summarize note"):
        document_text = extract_uploaded_document(uploaded_document) if uploaded_document else raw_text
        summary = summarize_document(document_text)
        st.text_area("Summary", summary, height=220)


with specialist_tab:
    st.subheader("Specialist appointment planning")
    body_part = st.selectbox(
        "Which area of the body is affected?",
        ["", "Head", "Shoulders", "Back", "Stomach"],
        format_func=lambda value: value or "Select a location",
    )
    feeling_type = st.selectbox("What type of feeling is it?", ["Pain", "Ache", "Soreness"])
    options = get_symptom_options(body_part.lower(), feeling_type.lower()) if body_part else []
    symptom_detail = st.selectbox(
        "Select the symptom that best matches the patient experience:",
        [""] + options,
        format_func=lambda value: value or "Select a symptom",
    )
    duration = st.text_input("How long has this symptom been present?", placeholder="For example: 3 days")
    severity = st.slider("Symptom severity", min_value=0, max_value=10, value=0, help="0 means not provided; 1 is mild and 10 is worst possible.")
    extra_info = st.text_area("Tell us more about the symptoms:", height=120)
    test_results = st.text_area("Review any test results or imaging notes:", height=120)

    if st.button("Recommend specialist and schedule"):
        follow_up_question = get_missing_information_question(
            symptom=symptom_detail,
            duration=duration,
            severity=severity,
            location=body_part,
        )

        if follow_up_question:
            st.warning(f"Sentient AI needs one more detail: {follow_up_question}")
            st.stop()

        plan = build_appointment_plan(
            body_part=body_part,
            feeling_type=feeling_type,
            symptom_detail=f"{symptom_detail} (severity {severity}/10, duration: {duration})",
            extra_info=extra_info,
            test_results=test_results,
        )

        checkpoint_dir = Path("models/specialty_classifier")
        if (checkpoint_dir / "labels.json").exists():
            completed_case = build_case_text(
                complaint=symptom_detail,
                symptoms=extra_info,
                location=body_part,
                duration=duration,
                severity=f"{severity}/10",
                test_results=test_results,
            )
            prediction = predict_specialty(completed_case, checkpoint_dir)
            st.success(
                f"Model specialty prediction: {prediction.specialty} "
                f"({prediction.probability:.1%} confidence)"
            )
            st.json(prediction.probabilities)
        else:
            st.success(f"Recommended specialist: {plan['specialist']}")
        st.write(plan["diagnosis_summary"])
        st.write("### Before appointment")
        for item in plan["before_appointment_instructions"]:
            st.write(f"- {item}")
        st.write("### Driving instructions")
        for item in plan["driving_instructions"]:
            st.write(f"- {item}")
        st.write("### After care")
        for item in plan["after_care_instructions"]:
            st.write(f"- {item}")
        st.write("### Additional medical diagnosis summary")
        st.write(summarize_diagnosis(f"{extra_info}\n{test_results}"))

with handoff_tab:
    st.subheader("Clinician handoff note")
    patient_name = st.text_input("Patient name")
    symptoms = st.text_area("Current symptoms")
    history = st.text_area("Past medical history")
    vitals = st.text_area("Vitals and observations")
    meds = st.text_area("Current medications")
    if st.button("Generate handoff"):
        note = build_handoff_note(patient_name, symptoms, history, vitals, meds)
        st.text_area("Handoff note", note, height=260)

st.markdown("---")
st.info("Urgent red flags: chest pain, trouble breathing, severe bleeding, confusion, fainting, or rapidly worsening symptoms should prompt immediate evaluation.")

