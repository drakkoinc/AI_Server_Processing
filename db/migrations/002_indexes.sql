-- 002_indexes.sql
-- Indexes and helper triggers for Drakko schema.
--
-- Conceptual purpose
-- ------------------
-- The schema stores the full AI output in JSONB (full_response) for auditability,
-- but we denormalize a few high-frequency filter fields into columns.
--
-- These indexes are a conservative default to support common UI queries like:
--   - "show me actionable emails" (is_actionable)
--   - "show me schedule&time emails" (major_category)
--   - "show me things that require a reply" (reply_required)
--   - grouping by thread
--
-- Add more indexes once you see real production query patterns.

BEGIN;

-- Helpful for list views / thread grouping
CREATE INDEX IF NOT EXISTS idx_email_messages_thread
  ON email_messages(provider_thread_id);

CREATE INDEX IF NOT EXISTS idx_email_messages_status
  ON email_messages(processing_status);

CREATE INDEX IF NOT EXISTS idx_email_messages_actionable
  ON email_messages(is_actionable);

-- Insights filtering
CREATE INDEX IF NOT EXISTS idx_email_insights_thread
  ON email_insights(provider_thread_id);

CREATE INDEX IF NOT EXISTS idx_email_insights_major_category
  ON email_insights(major_category);

CREATE INDEX IF NOT EXISTS idx_email_insights_sub_action_key
  ON email_insights(sub_action_key);

CREATE INDEX IF NOT EXISTS idx_email_insights_reply_required
  ON email_insights(reply_required);

-- Tasks indexes (optional)
CREATE INDEX IF NOT EXISTS idx_tasks_thread
  ON tasks(provider_thread_id);

CREATE INDEX IF NOT EXISTS idx_tasks_status
  ON tasks(status);

-- Auto-update updated_at on row changes (simple pattern)
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS trigger AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_email_messages_updated_at ON email_messages;
CREATE TRIGGER trg_email_messages_updated_at
  BEFORE UPDATE ON email_messages
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_tasks_updated_at ON tasks;
CREATE TRIGGER trg_tasks_updated_at
  BEFORE UPDATE ON tasks
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

COMMIT;
