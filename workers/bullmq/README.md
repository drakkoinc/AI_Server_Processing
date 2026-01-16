# BullMQ worker example (Node)

This folder is an **example** queue worker that matches the pipeline described in your docs.

- Node/Gmail poller inserts a row into `email_messages`
- Node/Gmail poller enqueues a job into Redis/BullMQ with `{ email_message_id }`
- This worker:
  1) fetches the raw provider payload from Postgres
  2) calls the AI server (`POST /v1/gmail/triage`)
  3) stores the AI response into `email_insights`
  4) marks `email_messages.processing_status = 'processed'`

## Quick start (local)

1) Start infra:

```bash
docker compose -f ../../infra/docker-compose.yml up -d
```

2) Install deps:

```bash
npm install
```

3) Set env vars:

```bash
export DATABASE_URL="postgres://drakko:drakko@localhost:5432/drakko"
export REDIS_URL="redis://localhost:6379"
export AI_SERVER_URL="http://localhost:8000"
export QUEUE_NAME="email_insights"
```

4) Run worker:

```bash
node worker.js
```

5) Enqueue a job (example):

```bash
node enqueue.js <email_message_uuid>
```

## Production notes

- Add retries + exponential backoff.
- Add dead-letter queue.
- Add request signing between the worker and AI server.
- Add idempotency: the example uses `ON CONFLICT` upserts.
