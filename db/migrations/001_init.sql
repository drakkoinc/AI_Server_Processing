-- 001_init.sql
-- Baseline schema for Drakko email ingestion + AI triage storage.
--
-- Conceptual model (why these tables exist)
-- ---------------------------------------
-- This schema supports the pipeline described in your PDFs:
--   (1) A poller ingests raw provider messages (Gmail JSON) into Postgres
--   (2) An AI service classifies / extracts triage signals from each message
--   (3) The triage output is stored for UI rendering, analytics, and reprocessing
--
-- Design goals
-- ------------
-- 1) Reprocessing-friendly: store the original provider payload (JSONB)
--    so you can re-run the model later without hitting Gmail again.
-- 2) Query-friendly: denormalize a few "hot" fields (major_category, sub_action_key,
--    reply_required, explicit_task) so your UI can filter/sort without JSONB scans.
-- 3) Minimal coupling: store the entire AI output as JSONB (`full_response`) so
--    you can evolve the model contract without constantly migrating tables.
--
-- NOTE:
-- - If you already have a schema, treat this file as a reference template.
-- - The worker example assumes these tables/columns exist.

BEGIN;

-- Needed for gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- -------------------------------------------------------------------------
-- Raw email messages (ingested by your Node poller)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS email_messages (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  provider text NOT NULL DEFAULT 'gmail',
  provider_message_id text NOT NULL,
  provider_thread_id text,

  history_id text,
  internal_date timestamptz,

  subject text,
  from_name text,
  from_email text,
  to_emails text[] DEFAULT ARRAY[]::text[],
  cc_emails text[] DEFAULT ARRAY[]::text[],

  label_ids text[] DEFAULT ARRAY[]::text[],
  snippet text,

  body_text text,
  body_html text,

  raw_payload jsonb NOT NULL,

  -- High-level flags derived from insights
  is_actionable boolean,
  processing_status text NOT NULL DEFAULT 'pending', -- pending|processed|ignored|error

  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),

  CONSTRAINT uq_email_messages_provider_msg UNIQUE (provider, provider_message_id)
);

-- -------------------------------------------------------------------------
-- Model outputs (1:1 with email_messages in this reference implementation)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS email_insights (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  email_message_id uuid NOT NULL REFERENCES email_messages(id) ON DELETE CASCADE,

  provider text NOT NULL,
  provider_message_id text NOT NULL,
  provider_thread_id text,

  -- Contract versioning + inference timestamp.
  -- The user-requested response JSON shape does not include these,
  -- but they are extremely useful for auditing and migrations.
  reference text NOT NULL DEFAULT 'drakko.gmail_triage.v2',
  predicted_at timestamptz NOT NULL DEFAULT now(),

  -- Denormalized triage fields (for fast UI filtering)
  major_category text NOT NULL,
  sub_action_key text NOT NULL,
  reply_required boolean NOT NULL,
  explicit_task boolean NOT NULL,
  task_type text,
  confidence real NOT NULL,
  reason text,

  -- Helpful structured blobs
  evidence jsonb,
  entities jsonb,

  -- The full response for auditability / reprocessing
  full_response jsonb NOT NULL,

  created_at timestamptz NOT NULL DEFAULT now(),

  CONSTRAINT uq_email_insights_email_message UNIQUE (email_message_id)
);

-- -------------------------------------------------------------------------
-- Tasks (optional): you may or may not use this in v2.
--
-- In the original v1 design, tasks were derived from a `suggestedActions[]` array.
-- The new v2 triage response is intentionally smaller and may not include full
-- task creation instructions.
--
-- Keep this table if you:
--   - already have a task system, or
--   - plan to extend the LLM output later with task payloads.
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tasks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  email_message_id uuid REFERENCES email_messages(id) ON DELETE SET NULL,
  provider_thread_id text,

  task_id_hint text NOT NULL,
  title text NOT NULL,
  instruction text,
  action_type text,
  priority text,

  due_at timestamptz,
  status text NOT NULL DEFAULT 'open', -- open|done|snoozed|cancelled

  metadata jsonb,

  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),

  CONSTRAINT uq_tasks_hint_per_thread UNIQUE (provider_thread_id, task_id_hint)
);

COMMIT;
