# Drakko Gmail Triage (AI server)

This repo is a **reference implementation** of an "email → intent → action" triage AI service.

It is designed to sit behind your Node/Gmail poller:

1. Node service polls Gmail every ~5 minutes and stores new messages in Postgres.
2. A trigger/worker enqueues a job (Redis/BullMQ, SQS, etc.).
3. A worker calls this AI server with the raw Gmail message JSON.
4. This AI server returns a strict JSON response that drives your UI and downstream automation.

This follows the "LLM as decision engine" framing from your PDFs:
- **What is this email about?** → `major_category`
- **What should I do?** → `sub_action_key` + `reply_required` / `explicit_task`
- **Why?** → `reason` + `evidence`
- **What objects matter?** → `entities`

## API

### Triage (primary)

`POST /v1/gmail/triage`

- Input: **raw Gmail Message JSON** (the same structure Gmail API returns)
- Output: a strict JSON object matching the requested schema

Example output (shape):

```json
{
  "message_id": "18c8a3f1b2d0a9a1",
  "thread_id": "18c89f3d1a2b4c55",
  "major_category": "schedule_and_time",
  "sub_action_key": "SCHEDULE_CONFIRM_TIME",
  "reply_required": true,
  "explicit_task": false,
  "task_type": null,
  "confidence": 0.87,
  "reason": "Sender asks you to confirm the meeting time.",
  "evidence": ["Can you confirm Friday at 2pm PT works?"],
  "entities": {
    "people": [{"email": "sarah@acme.com", "role": "sender"}],
    "dates": [{"text": "Friday 2pm PT", "iso": "2025-12-26T14:00:00-08:00", "type": "meeting_time"}],
    "money": [],
    "docs": [],
    "meeting": {"topic": "Contract approval timeline", "start_at": "2025-12-26T14:00:00-08:00", "tz": "America/Los_Angeles"}
  }
}
```

### Compatibility alias

`POST /v1/gmail/classify` → same behavior as `/v1/gmail/triage`.

## Run locally

### 1) Install deps

```bash
pip install -r requirements.txt
```

### 2) Set env vars

```bash
export OPENAI_API_KEY="..."
export LLM_PROVIDER="openai"
export OPENAI_MODEL="gpt-5.2"
```

### 3) Start server

```bash
uvicorn app.main:app --reload --port 8000
```

### 4) Test call

```bash
python scripts/call_api.py --url http://localhost:8000/v1/gmail/triage --input scripts/example_gmail_input.json
```

## Database schema + migrations

This repo includes reference SQL migrations under `db/migrations`.

For local development, we include `infra/docker-compose.yml` to spin up Postgres + Redis:

```bash
docker compose -f infra/docker-compose.yml up -d
```

Apply migrations:

```bash
docker exec -i drakko_postgres psql -U drakko -d drakko < db/migrations/001_init.sql
docker exec -i drakko_postgres psql -U drakko -d drakko < db/migrations/002_indexes.sql
```

## Queue worker example (BullMQ)

See `workers/bullmq/`.

This example worker:
- consumes `email_message_id` jobs from Redis
- fetches the raw Gmail message from Postgres
- calls this AI server
- writes the AI response into `email_insights`
- updates `email_messages.processing_status`

> In production you should add retries, dead-letter queues, and request signing.

## Project layout

- `app/models.py`         Pydantic schemas for request/response + Structured Outputs.
- `app/gmail.py`          Gmail MIME parsing + base64url decoding.
- `app/preprocess.py`     HTML → text, lightweight signal extraction (links, dates, money).
- `app/prompt.py`         Versioned prompt for triage output.
- `app/llm/*`             Provider clients (OpenAI + local stub).
- `app/pipeline.py`       Orchestration (parse → preprocess → LLM → postprocess).
- `app/postprocess.py`    Deterministic cleanup: IDs, date parsing, sender entity filling.
- `db/migrations/*`       Postgres schema migrations.
- `workers/bullmq/*`      Example queue worker (Node + BullMQ + Postgres).
