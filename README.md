# Sentient AI

## Clinical Specialist Matching and Scheduling Assistant

Sentient AI is an INAI 6986 capstone project built for Zoticus AI's agentic AI platform. It reads a de-identified clinical intake, recommends one of ten medical specialties for a nonemergency outpatient referral, and drafts an appointment for staff approval. Emergencies, low-confidence results and anything unusual go to clinical staff instead.

It is a decision-support prototype tested on synthetic data. It does not diagnose, prescribe, or confirm appointments, and no patient is contacted until a staff member approves the draft.

## Milestone 3 at a glance

| Part | What it is | Where |
| --- | --- | --- |
| Model service | FastAPI app that loads the frozen Milestone 2 model (MLP 64, run `mlp_64_v1`) and the deterministic safety gate | `sentient_api.py` |
| Frozen model | The scikit-learn pipeline frozen at the end of Milestone 2 | `experiments/milestone2_resub_v13_seed101/frozen_model.joblib` |
| Main workflow | n8n workflow: intake → validation → safety gate → Claude staff summary → specialty match → booking policy → draft or review | `n8n/Sentient AI - Milestone 3 Workflow.json` |
| Approval workflow | n8n workflow that emails the patient only after staff set `reviewer_decision = APPROVED` | `n8n/Sentient AI - Staff Approval Notifier.json` |
| Manual test plan | PowerShell request example, required scenarios and evidence table; execution results still need to be recorded | Testing instructions below |

## Run it from a fresh clone

### 1. Install and start the model service

```bash
git clone https://github.com/Sade421/Zoticus-AI.git
cd Zoticus-AI
pip install -r requirements.txt
uvicorn sentient_api:app --host 127.0.0.1 --port 8000
```

`scikit-learn` is pinned to 1.9.1 because the frozen model was saved with that version. The service refuses to start if the model's classes do not match the ten supported specialties.

The service URL must be reachable from the n8n runtime. For a Docker setup, verify host routing and the service bind address; Linux may require an explicit host-gateway mapping. For n8n Cloud, use a reachable hosted service URL rather than localhost or `host.docker.internal`.

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

1. In n8n, import `n8n/Sentient AI - Milestone 3 Workflow.json` and `n8n/Sentient AI - Staff Approval Notifier.json`.
2. Open the **Validate Intake** node in the main workflow and edit `CONFIG`:
   - `api_base_url`: where n8n reaches the service (default `http://host.docker.internal:8000`)
   - `staff_email`: the staff inbox for review and approval emails
   - `calendar_id`: the Google Calendar that holds draft appointments. Also verify the actual Calendar field in **Create Draft in Google Calendar**; a fixed node value is not changed by editing CONFIG.
   - `confidence_threshold`: 0.50 (see [Booking policy](#booking-policy))
3. Connect these n8n credentials: Anthropic (staff summary), Gmail (staff and patient emails), Google Calendar (drafts) and Google Sheets (review queue). No keys are stored in this repository.
4. Create a Google Sheet with a tab named `Review_Queue` and this header row, then select it in **Append to Review Queue** and in the approval workflow:

   ```text
   case_id, received_at, route, priority, reason, specialty_suggestion, confidence, ci95_low, ci95_high, clinical_summary, clarification_questions, status, proposed_start, calendar_event_id, calendar_link, patient_email, patient_first_name, assigned_to, reviewer_decision, reviewed_at, patient_notified_at
   ```

5. Activate both workflows.

### 3. Test manually with Windows PowerShell

There is currently no committed `tests/` folder or shell test runner. Use the PowerShell example below to send synthetic intake requests. The scenarios in the table are a test plan, not completed results.

Use an inbox you control for test notifications. Enabled nodes can create real draft calendar events, send staff notifications and write spreadsheet rows.

1. Start the model service and verify that n8n can reach it.
2. Save and activate/publish the main workflow, then copy the Webhook node's **Production URL**. Production runs appear under **Executions**.
3. Alternatively, copy the **Test URL** and start **Listen for test event** or **Execute workflow** before sending the request.
4. Keep the Webhook response mode set to **Using Respond to Webhook Node**.
5. Replace the URL below. This first example is intentionally short and should produce a review outcome, not a draft.

```powershell
$url = "PASTE_YOUR_WEBHOOK_URL_HERE"

$body = @{
    case_id = "M3-SHORT-" + (Get-Date -Format "yyyyMMdd-HHmmss")
    clinical_text = "Synthetic intake: knee discomfort for three weeks; requesting outpatient evaluation."
    patient_email = "test@example.com"
    patient_first_name = "Test"
    screening_complete = $true
}

function Send-SentientIntake {
    param([hashtable]$Intake)
    $json = $Intake | ConvertTo-Json -Depth 10
    try {
        $result = Invoke-WebRequest -Uri $url -Method Post -ContentType "application/json" -Body $json -UseBasicParsing
        Write-Output ("HTTP " + [int]$result.StatusCode)
        Write-Output $result.Content
    }
    catch {
        if ($null -ne $_.Exception.Response) {
            Write-Output ("HTTP " + [int]$_.Exception.Response.StatusCode)
        }
        if ($_.ErrorDetails.Message) {
            Write-Output $_.ErrorDetails.Message
        }
        else {
            Write-Output $_.Exception.Message
        }
    }
}

Send-SentientIntake -Intake $body
```

For each subsequent case, copy the body with `$case = $body.Clone()`, assign a new `case_id`, change the relevant fields, then run `Send-SentientIntake -Intake $case`. Preserve the exact request body with the run evidence. Reuse an ID only for the deliberate duplicate test.

| Case | Input or action | Expected behavior to verify |
| --- | --- | --- |
| T1 Eligible draft | Complete synthetic orthopedic intake of at least 200 characters, screening complete; actual matcher and summary outputs must satisfy all booking checks | Private DRAFT calendar event and queue row PENDING_APPROVAL; patient notice held |
| T2 Review | Vague nonemergency intake or an observed model confidence below 0.50 | Staff review without a draft; record whether the reason is clarification or low confidence |
| T3 Emergency | Synthetic intake containing an active emergency indicator, such as "CHIEF COMPLAINT: Chest pain now." | Emergency guidance returned before Gmail and Sheets; no summary, specialty matching or calendar event |
| T4 Incomplete screening | Set `screening_complete = $false` on otherwise valid intake | REVIEW_REQUIRED; no draft |
| T5 Invalid input | Remove email with `$case.Remove("patient_email")` | HTTP 400; no service call, notification or queue write |
| T6 Service outage | Stop only the model service, send a valid request, then restart the service | Caught connection failure and staff review; verify queue outcome |
| T7 Duplicate | Resend the exact T1 request with the same case_id | Document actual behavior; duplicate prevention is not implemented, so another draft/row may be created |
| T8 Short intake | Use fewer than 200 characters with otherwise passing summary and model checks | Staff review with the short-intake reason |
| T9 Abdominal–pelvic boundary | Use a synthetic intake whose actual top two model candidates both belong to GI / Nephrology / Urology / OB-GYN | Staff review with the boundary reason; retain top_candidates as evidence |

A case routed to review by an earlier check does not prove a later rule was exercised. For T8 and T9, verify the specific policy reason. Do not assume that mentioning GI or Urology in text guarantees those top candidates. Any mocked rule test must be labeled separately from an end-to-end model run.

For T3, inspect execution timing and confirm the dedicated emergency response node runs before notifications and logging. Later logging failures cannot be reported in an HTTP response that has already been sent; inspect the execution and queue for those outcomes.

### 4. Record execution evidence and staff approval

For every case, fill one row using actual execution results:

| Case ID | Execution ID | HTTP status | Safety status | Specialty | Confidence | Route/reason | Calendar event ID | Queue outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Not yet recorded | — | — | — | — | — | — | — | — |

Use N/A when matching or booking was skipped. Keep model suggestions separate from an approved referral. Retain the invalid-input response, outage error, duplicate outcome and emergency response timing.

After a successful synthetic draft, configure and activate the staff approval notifier. Use a controlled test inbox in that draft's row, set `reviewer_decision` to `APPROVED`, and verify one notification and the `patient_notified_at` timestamp. Record its execution ID. Uploading the notifier does not by itself demonstrate successful delivery.

No completed batch, approval-delivery result or new clean-clone result is claimed by these instructions. Add verified evidence after running the tests. Link the demonstration video in the report and Canvas. Also report the share of real-length evaluation intake that reaches a draft after the additional rules; this is separate from the synthetic routing tests.

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
