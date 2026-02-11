/**
 * BullMQ worker (Node) — reference implementation (v3 triage schema)
 *
 * Conceptual role in the overall Drakko pipeline
 * ----------------------------------------------
 * This worker is the "bridge" between:
 *   1) ingestion  (poll Gmail → write email_messages.raw_payload)
 *   2) inference  (call the AI server to classify/extract triage signals)
 *   3) persistence (store insights for UI + future automation)
 *
 * Why a queue worker at all?
 * --------------------------
 * If your poller called the AI server directly, it would:
 *   - block ingestion on model latency
 *   - make retries/error handling brittle
 *   - be hard to scale when many emails arrive at once
 *
 * BullMQ (Redis-backed) turns each email into a durable job.
 * Workers can scale horizontally and jobs can retry safely.
 *
 * Per job contract
 * ----------------
 * Job payload: { email_message_id: <uuid> }
 *
 * What this worker does (per job)
 * -------------------------------
 * 1) Fetch raw Gmail JSON from `email_messages.raw_payload`
 * 2) POST to AI server endpoint (`/rd/api/v1/ai/triage`)
 * 3) Upsert AI output into `email_insights` (JSONB + denormalized columns)
 * 4) Mark email_messages row as processed and set is_actionable
 *
 * Technical notes
 * --------------
 * - This is a readable reference worker. Production add-ons usually include:
 *   - exponential backoff retries
 *   - dead-letter queue
 *   - request signing worker ↔ AI server
 *   - concurrency limits / rate limiting
 *   - structured logging + tracing
 */

import { Worker } from "bullmq";
import IORedis from "ioredis";
import pg from "pg";

const { Pool } = pg;

// ---------------------------------------------------------------------------
// Env configuration
// ---------------------------------------------------------------------------
const REDIS_URL = process.env.REDIS_URL || "redis://localhost:6379";
const DATABASE_URL = process.env.DATABASE_URL || "postgres://drakko:drakko@localhost:5432/drakko";
const AI_SERVER_URL = process.env.AI_SERVER_URL || "http://localhost:8000";
const QUEUE_NAME = process.env.QUEUE_NAME || "email_insights";

// This reference string is stored in Postgres alongside the triage output.
// The AI server output JSON does not include `reference`, but keeping a version string
// in the DB is useful when you evolve the contract.
const CONTRACT_REFERENCE = process.env.CONTRACT_REFERENCE || "drakko.gmail_insights.v3";

// ---------------------------------------------------------------------------
// Shared connections
// ---------------------------------------------------------------------------
const redis = new IORedis(REDIS_URL, { maxRetriesPerRequest: null });
const pool = new Pool({ connectionString: DATABASE_URL });

async function fetchEmailMessage(emailMessageId) {
  const { rows } = await pool.query(
    `SELECT id, provider, provider_message_id, provider_thread_id, raw_payload
     FROM email_messages
     WHERE id = $1`,
    [emailMessageId]
  );
  if (rows.length === 0) return null;
  return rows[0];
}

async function callAiServer(rawPayload) {
  // Call the v3 triage endpoint.
  const r = await fetch(`${AI_SERVER_URL}/rd/api/v1/ai/triage`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(rawPayload),
  });

  const text = await r.text();
  if (!r.ok) {
    throw new Error(`AI server ${r.status}: ${text}`);
  }

  // v3 response is wrapped: { "output": { ... } }
  const envelope = JSON.parse(text);
  return envelope.output;
}

async function upsertInsights(emailMessageId, emailRow, output) {
  // Persist the triage result.
  //
  // Denormalized fields let your UI query quickly:
  // - major_category
  // - explicit_task
  // - sub_action_key
  //
  // v3 no longer has reply_required / task_type / reason as top-level fields.
  // We derive reply_required from urgency_signals and store the full output as JSONB.
  const replyRequired = output.urgency_signals?.urgency === "critical" ||
                         output.urgency_signals?.urgency === "high" ||
                         (output.suggested_reply_action?.length > 0);
  const taskType = output.task_proposal?.type ?? null;
  const reason = output.extracted_summary?.ask ?? null;

  await pool.query(
    `INSERT INTO email_insights (
        email_message_id,
        provider,
        provider_message_id,
        provider_thread_id,
        reference,
        predicted_at,
        major_category,
        sub_action_key,
        reply_required,
        explicit_task,
        task_type,
        confidence,
        reason,
        evidence,
        entities,
        full_response
      ) VALUES (
        $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16
      )
      ON CONFLICT (email_message_id) DO UPDATE SET
        reference = EXCLUDED.reference,
        predicted_at = EXCLUDED.predicted_at,
        major_category = EXCLUDED.major_category,
        sub_action_key = EXCLUDED.sub_action_key,
        reply_required = EXCLUDED.reply_required,
        explicit_task = EXCLUDED.explicit_task,
        task_type = EXCLUDED.task_type,
        confidence = EXCLUDED.confidence,
        reason = EXCLUDED.reason,
        evidence = EXCLUDED.evidence,
        entities = EXCLUDED.entities,
        full_response = EXCLUDED.full_response`,
    [
      emailMessageId,
      emailRow.provider,
      emailRow.provider_message_id,
      emailRow.provider_thread_id,
      CONTRACT_REFERENCE,
      output.debug?.analysis_timestamp ?? new Date().toISOString(),
      output.major_category,
      output.sub_action_key,
      replyRequired,
      output.explicit_task,
      taskType,
      output.confidence,
      reason,
      JSON.stringify(output.evidence ?? []),
      JSON.stringify(output.entities ?? null),
      JSON.stringify(output),
    ]
  );
}

async function markEmailProcessed(emailMessageId, output) {
  // Derive a simple is_actionable signal.
  // In v3 we check explicit_task, urgency, and whether reply actions were suggested.
  const isActionable = Boolean(
    output.explicit_task ||
    output.urgency_signals?.urgency === "critical" ||
    output.urgency_signals?.urgency === "high" ||
    (output.suggested_reply_action?.length > 0)
  );

  await pool.query(
    `UPDATE email_messages
     SET is_actionable = $2,
         processing_status = 'processed'
     WHERE id = $1`,
    [emailMessageId, isActionable]
  );
}

const worker = new Worker(
  QUEUE_NAME,
  async (job) => {
    const emailMessageId = job.data?.email_message_id || job.data?.emailMessageId;
    if (!emailMessageId) {
      throw new Error("Job missing email_message_id");
    }

    const emailRow = await fetchEmailMessage(emailMessageId);
    if (!emailRow) {
      throw new Error(`email_messages row not found for id=${emailMessageId}`);
    }

    // callAiServer returns the unwrapped output (not the envelope).
    const output = await callAiServer(emailRow.raw_payload);

    await upsertInsights(emailMessageId, emailRow, output);
    await markEmailProcessed(emailMessageId, output);

    return { major_category: output.major_category, sub_action_key: output.sub_action_key };
  },
  { connection: redis }
);

worker.on("completed", (job, result) => {
  console.log(`[completed] job=${job.id} result=${JSON.stringify(result)}`);
});

worker.on("failed", (job, err) => {
  console.error(`[failed] job=${job?.id} error=${err?.stack || err}`);
});
