# Database schema (reference)

This directory contains **reference SQL migrations** for a Drakko-like email triage pipeline.

- `email_messages` stores the raw provider payload (Gmail JSON) + a few normalized fields.
- `email_insights` stores the AI triage output (full JSON) + denormalized columns for fast filtering.
- `tasks` is optional and included as a placeholder if you later extend the model output to create tasks.

Apply migrations (local docker example):

```bash
docker compose -f infra/docker-compose.yml up -d

docker exec -i drakko_postgres psql -U drakko -d drakko < db/migrations/001_init.sql
docker exec -i drakko_postgres psql -U drakko -d drakko < db/migrations/002_indexes.sql
```

If you already have tables, treat these as templates and adapt column names/types.
