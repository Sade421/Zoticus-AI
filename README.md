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
- Prepared size reported by the latest supplied notebook: 773
- Supported specialties: 10
- Source: [MTSamples](https://www.mtsamples.com/)

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
9. Records fixed split assignments for reproducibility.

The latest supplied notebook also filters notes using a keyword-based multi-specialty heuristic, truncates text at the earliest recognized decision header, and drops text shorter than 40 characters. Notes without a recognized header remain in the current pipeline; they require a leakage audit. The heuristic does not establish clinically validated single-specialty ground truth.

Raw data remains unchanged. Cleaned and processed files are stored separately.

## Data Profile

The supplied `Sentient_AI_Data_Preparation (5).ipynb` reports the following stages. These values replace the earlier nine-label profile.  These values reflect a 10-specialty-label profile.

| Preparation stage | Records |
|---|---:|
| Source rows loaded | 4,999 |
| After selection of ten labels and removal of empty transcription text | 1,773 |
| After exact transcription duplicate removal | 1,692 |
| After keyword-based multi-specialty filtering | 1,552 |
| After pre-decision truncation and minimum-length filtering | 773 |
| Training partition (Section 7) | 525 |
| Validation partition (Section 7) | 132 |
| Test partition (Section 7) | 116 |

- Exact duplicates removed from the selected, nonempty subset: **81**.
- Records removed by the multi-specialty heuristic: **140**.
- Records removed for fewer than 40 characters of remaining text: **779**.
- Modeling input: `text`; target: `specialty`.
- Notes without a recognized decision header before minimum-length filtering: **255**.
- The near-duplicate stage reports **772 clusters**, **one pair at similarity ≥ 0.90**, and **20 review-band pairs**.

Section 6 reports a preliminary 463/155/155 split and zero cross-split duplicate pairs. Section 7 subsequently replaces it with the 525/132/116 split used downstream and includes assertions for disjoint cases and clusters. The preliminary audit should not be presented as a separate measured audit of the final split.

The 116-record test partition was reserved during revision from previously available development records. No test evaluation is shown in the latest supplied notebook. The earlier development history disclosed prior exposure of test cases, so this partition should not be represented as a fully independent final evaluation.

### Supported specialties and reported split counts

| Specialty label | Training | Validation | Test |
|---|---:|---:|---:|
| Cardiovascular / Pulmonary | 156 | 39 | 35 |
| Orthopedic | 100 | 25 | 22 |
| Gastroenterology | 53 | 13 | 12 |
| Obstetrics / Gynecology | 48 | 12 | 10 |
| Urology | 40 | 10 | 9 |
| Nephrology | 35 | 9 | 8 |
| Neurology | 34 | 9 | 7 |
| ENT - Otolaryngology | 25 | 6 | 6 |
| Ophthalmology | 22 | 6 | 5 |
| Dermatology | 12 | 3 | 2 |
| **Total** | **525** | **132** | **116** |

These are dataset labels. In particular, `Cardiovascular / Pulmonary` combines disciplines and does not identify one individual clinician or distinguish cardiology from pulmonology.

## Proposed Approach

The latest supplied notebook compares:

1. Majority-class and deterministic keyword baselines
2. Logistic Regression on TF-IDF features
3. MLP classifiers with hidden layers (64) and (64, 32) on TF-IDF features
4. A zero-shot Claude API pipeline (`zero_shot_v1`)

Claude is the leading candidate in the saved validation results. Final selection remains provisional pending reproducibility checks, leakage review, independent evaluation, and workflow safety testing.

The planned application output contains:

- Recommended specialist
- Confidence indicator
- Brief rationale
- Supporting evidence from the input
- Review-required status

The current Claude evaluation requests only a JSON specialty label. It does not measure confidence, rationale quality, supporting evidence, or review-required status. The intended evaluation policy excludes test records from training, prompt examples, model selection, and threshold tuning.

## Evaluation

### Primary metric

Top-1 specialist agreement: the proportion of cases whose single predicted specialty matches the reference label. The project target is at least 85% on an independent held-out evaluation. Current reported scores are validation results.

### Supporting metrics

- Macro F1 score
- Per-specialty precision and recall
- Confusion matrix
- Abstention or escalation rate
- Appointment-instruction safety pass rate
- Performance by symptom-description style

### Project acceptance targets

| Measure | Required threshold |
|---|---:|
| Specialist agreement | At least 85% |
| Instruction-safety pass rate | At least 90% |
| Cases producing a draft booking | At least 90% |

The workflow remains a demonstration if any required threshold is not met.

## Results and Evidence

### Saved validation results

Sections 9–12 and the Section 15 summary report the following scores on **132 validation cases** across ten labels. The two baselines are reported in Section 9; they are commented out of the Section 15 summary. Top-1 specialist agreement is the primary metric; macro F1 is a supporting metric. Scores are reproduced at the notebook's displayed precision.

| Model or baseline | Top-1 specialist agreement | Macro F1 |
|---|---:|---:|
| Majority class | 29.5% | 0.046 |
| Keyword baseline | 50.0% | 0.481 |
| Logistic Regression (TF-IDF) | 84.8% | 0.843 |
| MLP (64) | 87.1% | 0.832 |
| MLP (64, 32) | 84.8% | 0.813 |
| Claude pipeline (zero-shot, including fallback predictions) | **90.2%** | **0.889** |

Claude has the highest reported validation agreement and macro F1. Its validation agreement exceeds the numeric 85% target, but this does **not** demonstrate that the held-out target or project acceptance criteria have been met.

### Claude per-specialty validation results

These values are copied from Section 12's saved classification report, which displays per-class metrics to two decimal places.

| Specialty label | Precision | Recall | F1 | Validation support |
|---|---:|---:|---:|---:|
| Cardiovascular / Pulmonary | 0.97 | 0.90 | 0.93 | 39 |
| Dermatology | 1.00 | 1.00 | 1.00 | 3 |
| ENT - Otolaryngology | 0.71 | 0.83 | 0.77 | 6 |
| Gastroenterology | 0.80 | 0.92 | 0.86 | 13 |
| Nephrology | 0.71 | 0.56 | 0.62 | 9 |
| Neurology | 0.69 | 1.00 | 0.82 | 9 |
| Obstetrics / Gynecology | 1.00 | 1.00 | 1.00 | 12 |
| Ophthalmology | 1.00 | 1.00 | 1.00 | 6 |
| Orthopedic | 1.00 | 0.88 | 0.94 | 25 |
| Urology | 0.91 | 1.00 | 0.95 | 10 |

Nephrology has the lowest reported Claude recall (0.56). Dermatology's perfect score represents only three validation cases and should not be interpreted as established generalization.

### Evidence and reproducibility limits

Primary evidence: the supplied `Sentient_AI_Data_Preparation (5).ipynb`, inspected on October 7, 2026. Upload this notebook to the repository alongside the README so readers can inspect the updated outputs.

Historical reference only: [older preparation notebook at commit 9916ad5](https://github.com/Sade421/Zoticus-AI/blob/9916ad52bce4faa413b64ce2761b72881da04ef9/notebook/data_preparation). That older notebook reports nine labels and different metrics; it is not the evidence for the current tables.

| Claim | Evidence location in the notebook |
|---|---|
| Source load and preparation counts | Sections 1–5 |
| Near-duplicate grouping and preliminary audit | Section 6 |
| Final split counts and separation assertions | Section 7 |
| Full-text diagnostic comparator | Section 8 |
| Baseline and comparator reports | Sections 9–11 |
| Claude report and fallback warnings | Section 12 |
| Model comparison table | Section 15 |

- These are **saved notebook outputs**, inspected for this README update; the notebook was not rerun. The latest supplied notebook includes execution counts for the modeling cells. A clean end-to-end run and committed case-level exports are still needed to independently reproduce the reported scores.
- Section 12's code records `model_id = "claude-sonnet-5-5"`, while its introductory prose names Claude 3.5 Sonnet. The exact served model cannot be independently confirmed from the committed artifacts.
- One saved Claude warning shows a missing text block for `case_3762`. The pipeline uses the fallback label `Cardiovascular / Pulmonary`; the reported score therefore evaluates the pipeline including fallback behavior, rather than successful model responses alone.
- Section 8 reports full-text Logistic Regression at 86.4% agreement and 0.866 macro F1, compared with 84.8% and 0.843 on the truncated input. This diagnostic comparison is consistent with sensitivity to removed content; it does not prove all remaining leakage is eliminated.
- The truncation output reports recognized headers in 1,297 of 1,552 records, leaving 255 without a recognized header; the code retains them if they satisfy the minimum length. These notes and the 20 review-band pairs require documented review. A completed audit is not supplied.
- The notebook contains code to export prediction, summary, review-pair, and split-manifest files, but those generated files are not present in this repository snapshot. Case-level scores, fallback-adjusted scores, and final split audits cannot yet be independently recomputed from committed exports.
- Section 13's raw recall counts use `mlp_64.predict(Xval)` and therefore describe MLP (64), not Claude. Section 14's term weights and occlusion example describe Logistic Regression, not Claude. Its displayed true-class probability drops from 0.455 to 0.404 after term removal.
- Held-out test performance, confidence calibration, escalation rate, appointment-instruction safety, booking coverage, and performance by symptom-description style are **not reported**.

## Current Implementation Status

The repository currently contains the README, source CSV, and preparation notebook. Updated model comparison results are available in the supplied notebook and must be uploaded to GitHub with this README. The n8n booking workflow and application-level confidence, evidence, and review outputs described below are planned capabilities; their implementation and acceptance results are not established by the committed files.

## Human Review and Safety

Patient safety and human oversight are required.

The workflow sends a case to clinical review when:

- Model confidence is below the approved threshold.
- The recommendation lacks supporting evidence.
- The output does not match a supported specialty.
- Appointment instructions fail the safety checklist.
- The input is incomplete or ambiguous.

Generated recommendations and instructions are drafts. Clinical staff retains final authority.

## Planned System Workflow

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
- scikit-learn for evaluation metrics
- Git and GitHub for version control

## Planned Repository Structure

The tree below describes the intended layout. The current committed paths are `README.md`, `mtsamples-v3.csv`, and `notebook/data_preparation` (notebook JSON stored without an `.ipynb` extension).

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

