# Sentient AI

## Clinical Specialist Matching and Scheduling Assistant

Sentient AI is a capstone project developed for Zoticus AI’s Agentic AI platform. The system analyzes de-identified clinical transcription text and recommends the appropriate medical specialty for a nonemergency outpatient referral.

The project also demonstrates an n8n workflow that drafts an appointment booking when the recommendation meets the confidence threshold. Low-confidence cases are sent to clinical staff for manual review.

## Project Decision

The system addresses one decision:

> Which medical specialist should receive the patient’s referral based on the available clinical narrative?

The system supports referral decisions but does not diagnose patients, prescribe treatment, or autonomously schedule appointments.

## Project Objectives

- Achieve at least 85% agreement between the recommended specialist and the held-out MTSamples specialty label.
- Achieve at least a 90% safety-review pass rate for generated appointment instructions.
- Produce a specialist recommendation and draft booking for at least 90% of test cases.
- Escalate low-confidence or unsupported recommendations to human review.

## Data Source

The primary dataset is the public MTSamples medical transcription corpus.

- Unit of observation: one medical transcription
- Input: de-identified transcription text
- Target: medical-specialty label
- Additional fields: description, sample type, and keywords when available
- Approximate original size: 5,000 transcription samples
- Final cleaned size: **[ADD AFTER PROFILING]**
- Supported specialties: **[ADD FINAL SPECIALTY LIST]**
- Source: [MTSamples](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)

The specialty label is treated as an evaluation benchmark. It is not assumed to be a clinically validated referral outcome.

## Data Preparation

The preparation pipeline performs the following steps:

1. Loads the source data.
2. Removes empty transcription records.
3. Removes duplicate or near-duplicate records.
4. Normalizes specialty labels.
5. Checks the text for possible identifying information.
6. Reviews the specialty distribution.
7. Selects specialties with sufficient examples.
8. Creates stratified training, validation, and held-out test sets.
9. Freezes the test set before prompt development.

Raw data remains unchanged. Cleaned and processed files are stored separately.

## Data Profile

Replace this table with your actual results after running the preparation notebook.

| Measure | Result |
|---|---:|
| Original rows | [ADD] |
| Rows after cleaning | [ADD] |
| Columns used | [ADD] |
| Duplicate records removed | [ADD] |
| Missing transcription records | [ADD] |
| Number of supported specialties | [ADD] |
| Training records | [ADD] |
| Validation records | [ADD] |
| Test records | [ADD] |

## Proposed Approach

The project compares:

1. A deterministic keyword-matching baseline
2. LLM prompt or pipeline variant A
3. LLM prompt or pipeline variant B

The LLM returns a structured result containing:

- Recommended specialist
- Confidence indicator
- Brief rationale
- Supporting evidence from the input
- Review-required status

The LLM does not train directly on the held-out test set.

## Evaluation

### Primary metric

Top-1 specialist agreement on the held-out test set.

### Supporting metrics

- Macro F1 score
- Per-specialty precision and recall
- Confusion matrix
- Abstention or escalation rate
- Appointment-instruction safety pass rate
- Performance by symptom-description style

### Deployment thresholds

| Measure | Required threshold |
|---|---:|
| Specialist agreement | At least 85% |
| Instruction-safety pass rate | At least 90% |
| Cases producing a draft booking | At least 90% |

The workflow remains a demonstration if any required threshold is not met.

## Human Review and Safety

Patient safety and human oversight are required.

The workflow sends a case to clinical review when:

- Model confidence is below the approved threshold.
- The recommendation lacks supporting evidence.
- The output does not match a supported specialty.
- Appointment instructions fail the safety checklist.
- The input is incomplete or ambiguous.

Generated recommendations and instructions are drafts. Clinical staff retain final authority.

## System Workflow

1. Patient intake or clinical text becomes available.
2. n8n sends the de-identified text to the specialist-matching pipeline.
3. The pipeline returns a specialist recommendation and confidence result.
4. High-confidence cases generate a draft booking for staff approval.
5. Low-confidence cases enter the clinician-review queue.
6. The system logs the recommendation, review status, and final action.

## Technology Stack

- Python
- Pandas
- Claude API
- n8n
- Streamlit or React for the prototype interface
- FastAPI for API endpoints
- scikit-learn for evaluation metrics
- Git and GitHub for version control

## Repository Structure

```text
healthcare_assistant/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── README.md
│   └── processed/
├── notebooks/
│   └── data_preparation.ipynb
├── src/
│   ├── preprocessing.py
│   ├── keyword_baseline.py
│   ├── specialist_matcher.py
│   └── evaluation.py
├── workflows/
│   └── specialist_matching_n8n.json
├── tests/
└── results/
    └── evaluation_summary.md
