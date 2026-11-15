# Sentient AI

## Clinical Specialist Matching and Scheduling Assistant

Sentient AI is an INAI 6986 capstone project built for Zoticus AI's agentic AI platform. It reads a de-identified clinical intake, recommends one of ten medical specialties for a nonemergency outpatient referral, and drafts an appointment for staff approval. Emergencies, low-confidence results and anything unusual go to clinical staff instead.

It is a decision-support prototype tested on synthetic data. It does not diagnose, prescribe, or confirm appointments, and no patient is contacted until a staff member approves the draft.

## Milestone 3 at a glance

| Part | What it is | Where |
| --- | --- | --- |
| Model service | FastAPI app that loads the frozen Milestone 2 model (MLP 64, run `mlp_64_v1`) and the deterministic safety gate | `sentient_api.py` |
| Frozen model | The scikit-learn pipeline frozen at the end of Milestone 2 | `experiments/milestone2_resub_v13_seed101/frozen_model.joblib` |
| Main workflow | n8n workflow: intake → validation → safety gate → Claude staff summary → specialty match → booking policy → draft or review | `n8n/Sentient AI - Clinicial Specialist Matching and Scheduling Assistant.json` |
| Approval workflow | n8n workflow that emails the patient only after staff set `reviewer_decision = APPROVED` | `n8n/Sentient AI - Staff Approval Notifier.json` |
| Test batch | Seven synthetic intakes covering every route, plus a script to send them | `tests/` |

## Run it from a fresh clone

### 1. Install and start the model service

```bash
git clone https://github.com/Sade421/Zoticus-AI.git
cd Zoticus-AI
pip install -r requirements.txt
uvicorn sentient_api:app --host 127.0.0.1 --port 8000
```

`scikit-learn` is pinned to 1.9.1 because the frozen model was saved with that version. The service refuses to start if the model's classes do not match the ten supported specialties.

If n8n runs in Docker on Linux, start the service with `--host 0.0.0.0` so the container can reach it through `host.docker.internal`. Docker Desktop on Mac and Windows works with the default host.

Check it:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","model_run_id":"mlp_64_v1","labels":[...10 specialties...]}
```

Also run the same check from inside the n8n container against the `api_base_url` you configure in step 2:

```bash
docker exec -it <n8n-container> wget -qO- http://host.docker.internal:8000/health
```

### 2. Import and configure the n8n workflows

1. In n8n, import both files from `n8n/`.
2. Open the **Validate Intake** node in the main workflow and edit `CONFIG`:
   - `api_base_url`: where n8n reaches the service (default `http://host.docker.internal:8000`)
   - `staff_email`: the staff inbox for review and approval emails
   - `calendar_id`: the Google Calendar that holds draft appointments
   - `confidence_threshold`: 0.50 (see [Booking policy](#booking-policy))
3. Connect these n8n credentials: Anthropic (staff summary), Gmail (staff and patient emails), Google Calendar (drafts) and Google Sheets (review queue). No keys are stored in this repository.
4. Create a Google Sheet with a tab named `Review_Queue` and this header row, then select it in **Append to Review Queue** and in the approval workflow:

   ```text
   case_id, received_at, route, priority, reason, specialty_suggestion, confidence, ci95_low, ci95_high, clinical_summary, clarification_questions, status, proposed_start, calendar_event_id, calendar_link, patient_email, patient_first_name, assigned_to, reviewer_decision, reviewed_at, patient_notified_at
   ```

5. Activate both workflows.

### 3. Run the test batch

With both workflows active and the service running:

```bash
chmod +x tests/send_intake.sh
WEBHOOK_URL=http://localhost:5678/webhook/sentient-ai-intake ./tests/send_intake.sh tests/intakes/0[1-5]*.json
```

Then stop the service (Ctrl+C in the uvicorn terminal) and send case 6, restart it, and send case 7:

```bash
./tests/send_intake.sh tests/intakes/06_service_stopped.json
./tests/send_intake.sh tests/intakes/07_duplicate_case_id.json
```

Finally, set `reviewer_decision` to `APPROVED` on case `M3-TEST-01`'s row in the sheet and confirm the approval workflow sends the patient notice and fills `patient_notified_at`.

| File | Case | Expected result |
| --- | --- | --- |
| `01_eligible_draft.json` | Full orthopedic referral, screening complete | HTTP 200, private DRAFT calendar event, queue row `PENDING_APPROVAL`, patient notice held |
| `02_low_confidence_review.json` | Vague intake with no clear specialty | HTTP 200, review row, no draft (reason: clarification needed or confidence below 0.50) |
| `03_emergency.json` | Chest pain with emergency indicators | HTTP 200, `EMERGENCY_ESCALATED`, URGENT staff email, no matching or draft |
| `04_incomplete_screening.json` | Same text as case 1, `screening_complete: false` | HTTP 200, `REVIEW_REQUIRED`, no draft |
| `05_missing_email_400.json` | No `patient_email` | HTTP 400 listing the missing field; nothing queued or emailed |
| `06_service_stopped.json` | Valid case sent while the service is down | HTTP 200, review row with priority HIGH and the connection error in the reason |
| `07_duplicate_case_id.json` | Case 1 sent again | A second draft and row are created (known gap: no duplicate prevention) |

Checked against the service before committing: case 1 is matched as Orthopedic at 0.97 confidence, case 2's top prediction is 0.26, and case 3 returns `EMERGENCY_ESCALATED`.

## Service API

All requests are JSON. The service receives de-identified clinical text only; it never receives patient names or email addresses and keeps no request logs.

### `GET /health`

Returns `status`, `model_run_id` (`mlp_64_v1`) and the ten supported labels.

### `POST /safety-gate`

Deterministic screening ported from the Milestone 2 notebook. No model and no LLM.

```json
{"case_id": "M3-TEST-01", "text": "...", "screening_complete": true}
```

Returns `safety_status`:

- `EMERGENCY_ESCALATED`: a red-flag phrase (for example chest pain, difficulty breathing, sudden one-sided weakness) or a dangerous vital sign (BP systolic under 100, SpO2 under 90%, RR over 22). The response includes emergency guidance.
- `REVIEW_REQUIRED`: screening is incomplete, a red flag appears in a negated or uncertain context, or vitals are present that need a reviewer.
- `NONEMERGENCY_ELIGIBLE`: none of the above.

### `POST /match`

```json
{"case_id": "M3-TEST-01", "text": "..."}
```

Applies the Milestone 2 pre-decision truncation (text is cut at the first decision header such as ASSESSMENT or PLAN), then runs the frozen model. Text shorter than 40 characters after truncation returns `match_status: REVIEW_REQUIRED`. Otherwise it returns `match_status: MATCHED`, `specialty`, `confidence`, `top_candidates` (top three with probabilities), `truncated`, `cut_header` and `model_run_id`.

## Booking policy

The **Apply Booking Policy** node allows a draft only when every check passes. The first check that fails becomes the review reason, and no default specialty is ever substituted.

1. Intake is valid, `screening_complete` is true, and the safety gate returned `NONEMERGENCY_ELIGIBLE`.
2. The Claude staff summary is present, well formed and has no open clarification questions.
3. `/match` returned `MATCHED` with one of the ten supported labels.
4. Confidence is at least **0.50**. On the Milestone 2 holdout this released 91 of 108 full notes, 92% of them correct.
5. The clinical text is at least **200 characters** (Milestone 2 commitment).
6. The top two candidates are not both in the abdominal–pelvic group (Gastroenterology, Nephrology, Urology, Obstetrics / Gynecology), the boundary identified in the Milestone 2 error analysis.
7. The prediction is not the combined `Cardiovascular / Pulmonary` label, which always needs staff to choose Cardiology or Pulmonology.

An allowed case gets a proposed 30-minute slot (the preferred date if it is a valid future weekday, otherwise the next business day, at 9:00 America/Chicago) and a private Google Calendar event marked DRAFT. Every route (draft, review and emergency) sends a staff email and writes a row to `Review_Queue`. The patient is emailed only by the approval workflow, after staff approve.

## Model

| Item | Value |
| --- | --- |
| Model | MLP with one hidden layer of 64 units on TF-IDF features (`mlp_64_v1`) |
| Artifact | `experiments/milestone2_resub_v13_seed101/frozen_model.joblib` |
| SHA-256 | `b38c66e021de4a9cc080033e2d6ce762a8bf7bb7613ad43ee6241aff7fb08a8f` |
| Development holdout | 92 of 108 correct (Milestone 2) |
| Labels | Cardiovascular / Pulmonary, Orthopedic, Gastroenterology, Neurology, Obstetrics / Gynecology, Urology, ENT - Otolaryngology, Nephrology, Dermatology, Ophthalmology |

`sentient_api.py` loads this file directly. It replaces the `frozen_predictor.py` call planned in Milestone 2; the model is the same, only the calling code differs. Set `SENTIENT_MODEL_PATH` to load the model from another location.

## Known limits

- **No duplicate prevention.** A repeated `case_id` creates a second draft, email and queue row.
- **No calendar availability check.** Every draft for a given day is proposed at 9:00.
- **Webhook has no authentication.** Add header authentication before exposing it beyond a local machine.
- **De-identification is assumed.** The workflow does not strip identifiers from the clinical text.
- **Emergency and review rows carry no patient contact.** Staff must look up the patient elsewhere to follow up.
- **Synthetic data only.** The threshold and screening rules are not clinically validated.

## Repository layout

```text
Zoticus-AI/
├── README.md
├── requirements.txt
├── sentient_api.py                     # Milestone 3 model service
├── n8n/                                # Milestone 3 workflow exports
├── tests/
│   ├── send_intake.sh                  # posts intake files to the webhook
│   └── intakes/                        # seven synthetic test cases
├── notebook/
│   ├── Milestone 1.ipynb               # data preparation
│   └── Sentient_AI_Milestone_2.ipynb   # model selection, safety gate, freezing
├── experiments/
│   └── milestone2_resub_v13_seed101/   # frozen model, splits, metrics, run registry
└── artifacts/                          # source data and audit files
```

## Earlier milestones

- **Milestone 1 (data):** MTSamples medical transcriptions filtered to ten specialty labels, deduplicated, truncated before decision headers, with a 40-character minimum. See `notebook/Milestone 1.ipynb`.
- **Milestone 2 (model):** baselines, logistic regression and MLP candidates compared on fixed splits; MLP 64 frozen with its safety gate and automation contract. See `notebook/Sentient_AI_Milestone_2.ipynb` and `experiments/milestone2_resub_v13_seed101/`.

Data source: [MTSamples](https://www.mtsamples.com/). The specialty labels are an evaluation benchmark, not clinically validated referral outcomes.
