/**
 * enqueue.js — helper script for local testing
 *
 * Conceptual purpose
 * ------------------
 * In production, your Gmail poller (or a DB trigger) would enqueue jobs automatically
 * when new emails are written into `email_messages`.
 *
 * For local dev, it's convenient to enqueue a job manually given an email_messages.id UUID.
 *
 * Usage
 * -----
 *   node enqueue.js <email_message_uuid>
 *
 * Environment
 * -----------
 *   REDIS_URL   (default: redis://localhost:6379)
 *   QUEUE_NAME  (default: email_insights)
 */

import { Queue } from "bullmq";
import IORedis from "ioredis";

// Keep these consistent with worker.js
const REDIS_URL = process.env.REDIS_URL || "redis://localhost:6379";
const QUEUE_NAME = process.env.QUEUE_NAME || "email_insights";

const emailMessageId = process.argv[2];
if (!emailMessageId) {
  console.error("Usage: node enqueue.js <email_message_uuid>");
  process.exit(1);
}

// BullMQ Queue is just a job "mailbox". The worker.js process will consume these jobs.
const redis = new IORedis(REDIS_URL, { maxRetriesPerRequest: null });
const queue = new Queue(QUEUE_NAME, { connection: redis });

// Job name is arbitrary; queue name is what matters. We keep both as "email_insights".
// The worker accepts `email_message_id` as the canonical key.
await queue.add("email_insights", { email_message_id: emailMessageId });

console.log(`Enqueued job for email_message_id=${emailMessageId}`);

await queue.close();
await redis.quit();
