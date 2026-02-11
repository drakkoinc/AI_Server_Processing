# Drakko Email AI Server

An email triage AI service that accepts raw Gmail API JSON, classifies messages into semantic categories, extracts entities, detects urgency and deadlines, proposes trackable tasks, and recommends UI actions — all through a single API call.

**Version:** 3.0.0 | **Schema:** v3 | **Contract:** `drakko.gmail_insights.v3`

## How It Works

Drakko sits behind your Gmail poller as the decision engine:

1. A Node service polls Gmail and stores new messages in Postgres.
2. A BullMQ worker picks up the job from Redis.
3. The worker calls Drakko's triage endpoint with the raw Gmail message JSON.
4. Drakko returns a rich JSON response that drives your UI and downstream automation.

The pipeline processes each email through five stages:

- **Parse** — Decode the nested MIME tree (base64url bodies, headers, parts) into a normalized structure.
- **Preprocess** — Extract observable signals: URLs, monetary expressions, time/deadline phrases.
- **Infer** — Send the structured context to OpenAI with a Pydantic-derived JSON schema (Structured Outputs).
- **Postprocess** — Apply deterministic server-side fixes: confidence clamping, deadline-to-ISO parsing, entity backfilling, debug metadata injection.
- **Respond** — Return the final `TriageResponse` envelope.

## API Endpoints

All endpoints are prefixed with `/rd/api/v1`.

### `POST /rd/api/v1/ai/triage`

The primary endpoint. Accepts raw Gmail message JSON and returns the full triage output.

**Input:** Raw Gmail `Message` JSON (the same structure the Gmail API returns).

**Output:** A `{ "output": { ... } }` envelope containing:

```json
{
  "output": {
    "major_category": "schedule_and_time",
    "sub_action_key": "SCHEDULE_CONFIRM_TIME",
    "explicit_task": false,
    "confidence": 0.91,
    "suggested_reply_action": [
      "Yes, Friday at 2 PM PT works for me.",
      "Can we push to Monday instead?"
    ],
    "task_proposal": {
      "type": "confirm_meeting",
      "title": "Confirm meeting time with Sarah",
      "description": "Sender asks you to confirm Friday 2 PM PT for the contract review.",
      "priority": "high",
      "status": "open",
      "scheduled_for": null,
      "due_at": "2026-02-13T14:00:00-08:00",
      "waiting_on": null
    },
    "recommended_actions": [
      { "key": "reply_confirm", "label": "Confirm Time", "kind": "PRIMARY", "rank": 1 },
      { "key": "reply_reschedule", "label": "Propose New Time", "kind": "SECONDARY", "rank": 2 },
      { "key": "snooze_1h", "label": "Snooze 1 Hour", "kind": "SECONDARY", "rank": 3 }
    ],
    "urgency_signals": {
      "urgency": "high",
      "deadline_detected": true,
      "deadline_text": "Friday 2pm PT",
      "reply_by": "2026-02-13T14:00:00-08:00",
      "reason": "Sender is waiting on your confirmation for a scheduled meeting."
    },
    "extracted_summary": {
      "ask": "Confirm whether Friday at 2 PM PT works for the contract review meeting.",
      "success_criteria": "Reply confirming or proposing an alternative time.",
      "missing_info": []
    },
    "entities": {
      "people": [
        { "email": "sarah@acme.com", "role": "sender" }
      ],
      "dates": [
        { "text": "Friday 2pm PT", "iso": "2026-02-13T14:00:00-08:00", "type": "meeting_time" }
      ],
      "money": [],
      "docs": [],
      "meeting": {
        "topic": "Contract approval timeline",
        "start_at": "2026-02-13T14:00:00-08:00",
        "tz": "America/Los_Angeles"
      }
    },
    "evidence": [
      "Can you confirm Friday at 2pm PT works?"
    ],
    "debug": {
      "analysis_timestamp": "2026-02-10T12:00:00+00:00",
      "model_version": "drakko-email-v3",
      "prompt_version": "triage-v3-2026-02"
    }
  }
}
```

### `GET /rd/api/v1/apidata`

Returns API metadata: name, version, schema version, contract reference, a listing of all endpoints with methods/paths/descriptions, and the full list of `major_category` values.

### `GET /rd/api/v1/health`

Returns health diagnostics: status (`healthy` or `degraded` based on 5-minute error rate), uptime, server start time, LLM provider check, Python version, request counts, and the last 10 errors.

### `GET /rd/api/v1/ai`

Returns AI model configuration: provider, model, temperature, timeout, prompt version, model version, capabilities list, and request counts.

## Output Schema

The triage output contains these top-level blocks:

| Field | Type | Description |
|-------|------|-------------|
| `major_category` | enum (11 values) | Semantic classification bucket |
| `sub_action_key` | string | SCREAMING_SNAKE_CASE action identifier |
| `explicit_task` | boolean | Whether the email contains a clear, actionable task |
| `confidence` | float [0.0, 1.0] | Model confidence in the classification |
| `suggested_reply_action` | string[] (0-3) | Quick-reply chips for the UI |
| `task_proposal` | object or null | Fully-formed task ready for a task manager |
| `recommended_actions` | object[] (0-4) | Ranked UI action buttons (PRIMARY/SECONDARY/DANGER) |
| `urgency_signals` | object | Urgency level, deadline detection, reply-by datetime |
| `extracted_summary` | object | Executive summary: ask, success_criteria, missing_info |
| `entities` | object | People, dates, money, documents, meetings |
| `evidence` | string[] (1-3) | Verbatim snippets from the email body |
| `debug` | object | Server-injected metadata: timestamp, model version, prompt version |

### Major Categories

`core_communication`, `decisions_and_approvals`, `schedule_and_time`, `documents_and_review`, `financial_and_admin`, `people_and_process`, `information_and_org`, `learning_and_awareness`, `social_and_people`, `meta_and_systems`, `other`

## Run Locally

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Set environment variables

```bash
export OPENAI_API_KEY="..."
export LLM_PROVIDER="openai"          # or "local"
export OPENAI_MODEL="gpt-5.2"
```

Optional overrides:

```bash
export LLM_TEMPERATURE="0.2"
export LLM_TIMEOUT_S="30"
export MAX_BODY_CHARS="12000"
export MODEL_VERSION="drakko-email-v3"
export CONTRACT_REFERENCE="drakko.gmail_insights.v3"
```

### 3) Start the server

```bash
uvicorn app.main:app --reload --port 8000
```

### 4) Test

```bash
# Check the API is running
curl http://localhost:8000/rd/api/v1/health

# View API metadata
curl http://localhost:8000/rd/api/v1/apidata

# View AI configuration
curl http://localhost:8000/rd/api/v1/ai

# Triage an email
curl -X POST http://localhost:8000/rd/api/v1/ai/triage \
  -H "Content-Type: application/json" \
  -d @scripts/example_gmail_input.json
```

### 5) Batch test (optional)

```bash
python scripts/batch_call_api.py \
  --url http://localhost:8000/rd/api/v1/ai/triage \
  --input scripts/example_gmail_input.json \
  --concurrency 3
```

## Database

Reference PostgreSQL schema is in `db/migrations/`. Spin up local Postgres + Redis:

```bash
docker compose -f infra/docker-compose.yml up -d
```

Apply migrations:

```bash
docker exec -i drakko_postgres psql -U drakko -d drakko < db/migrations/001_init.sql
docker exec -i drakko_postgres psql -U drakko -d drakko < db/migrations/002_indexes.sql
```

Three tables:
- **`email_messages`** — Raw ingested emails with the full Gmail JSON in `raw_payload` (JSONB) for reprocessing.
- **`email_insights`** — AI triage outputs with denormalized hot fields for fast UI filtering plus `full_response` (JSONB) for auditability.
- **`tasks`** — Optional task storage derived from triage outputs.

## Queue Worker (BullMQ)

See `workers/bullmq/`. The worker consumes `email_message_id` jobs from Redis, fetches raw Gmail JSON from Postgres, calls the AI server, persists insights, and marks emails as processed.

```bash
# Start the worker
node workers/bullmq/worker.js

# Enqueue a test job
node workers/bullmq/enqueue.js <email_message_uuid>
```

> In production, add retries, dead-letter queues, and request signing.

## Docker

```bash
docker build -t drakko-ai .
docker run -p 8000:8000 -e OPENAI_API_KEY="..." drakko-ai
```

## Project Layout

```
app/
  config.py            Centralized configuration from environment variables.
  models.py            Pydantic schemas: Gmail input, LLM output, API response.
  prompt.py            Versioned system prompt + category/action-key taxonomy guides.
  gmail.py             Gmail MIME parsing: base64url decoding, header extraction, body resolution.
  preprocess.py        HTML-to-text, signal extraction (links, money, time phrases).
  postprocess.py       Deterministic cleanup: confidence clamping, datetime parsing,
                       entity backfilling, urgency normalization, debug injection.
  pipeline.py          Orchestration: parse -> preprocess -> LLM -> postprocess.
  main.py              FastAPI app with 4 endpoints + health/error tracking.
  llm/
    openai_client.py   OpenAI Structured Outputs adapter (Pydantic-parsed responses).
    local_client.py    Stub for a future locally-hosted model.

db/migrations/
  001_init.sql         Baseline Postgres schema (email_messages, email_insights, tasks).
  002_indexes.sql      Indexes for common UI queries + updated_at triggers.

workers/bullmq/
  worker.js            BullMQ worker: fetch email -> call AI -> persist insights.
  enqueue.js           CLI helper for manual job enqueueing during local dev.

scripts/
  batch_call_api.py    Async batch tester with concurrency control and distribution reporting.

tests/
  test_schema.py       Integration tests: Gmail parsing + postprocessing validation.
  test_preprocess.py   Unit tests: HTML-to-text, link/money/time extraction.

infra/
  docker-compose.yml   Local Postgres 16 + Redis 7.

Dockerfile             Python 3.11-slim, uvicorn, port 8000.
requirements.txt       Python dependencies.
```

## Tech Stack

- **Python 3.11** + **FastAPI** + **Pydantic v2** — API server and schema validation
- **OpenAI Structured Outputs** (GPT-5.2) — Forces LLM responses to match exact Pydantic schemas
- **PostgreSQL 16** — Email storage, AI output persistence, task tracking
- **Redis 7** + **BullMQ** (Node.js) — Async job queue for email processing workers
- **BeautifulSoup** + **lxml** — HTML email body parsing
- **python-dateutil** — Timezone-aware datetime parsing for deadlines and meeting times
- **Docker** — Containerized deployment
