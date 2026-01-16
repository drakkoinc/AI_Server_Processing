#!/usr/bin/env python3
"""
Batch test runner for Drakko Gmail triage API.

Conceptually:
- Your API endpoint (/v1/gmail/triage) accepts ONE Gmail message JSON object per request.
- The sample file contains MANY messages (a JSON array).
- This script iterates over the list, posts each message to the API, and:
  1) validates the response shape against EmailTriageResponse
  2) writes results to a JSONL file (one response per line)
  3) writes failures to a separate JSONL file
  4) prints a category/action-key distribution summary

Technically:
- Uses httpx.AsyncClient for simple concurrency.
- Uses app.models.EmailTriageResponse to ensure schema correctness.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

# IMPORTANT:
# Run this script from the repo root so `app.*` imports resolve cleanly.
from app.models import EmailTriageResponse


@dataclass
class BatchResult:
    ok: bool
    input_message_id: Optional[str]
    status_code: Optional[int]
    response: Optional[Dict[str, Any]]
    error: Optional[str]


def load_messages(path: Path) -> List[Dict[str, Any]]:
    """
    Loads either:
      - a JSON array of Gmail messages (your sample file format), OR
      - a single JSON object (wraps into a list)
    """
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Empty input file: {path}")

    data = json.loads(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]

    raise TypeError(f"Unexpected JSON top-level type: {type(data)}")


async def post_one(
    client: httpx.AsyncClient,
    url: str,
    msg: Dict[str, Any],
    timeout_s: float,
) -> BatchResult:
    """
    POST one Gmail message to the triage endpoint, validate response schema.
    """
    input_message_id = msg.get("id")  # Gmail message id
    try:
        r = await client.post(url, json=msg, timeout=timeout_s)
        status = r.status_code

        if status >= 400:
            return BatchResult(
                ok=False,
                input_message_id=input_message_id,
                status_code=status,
                response=None,
                error=f"HTTP {status}: {r.text[:4000]}",
            )

        data = r.json()

        # Schema validation: raises if fields/types don't match
        EmailTriageResponse.model_validate(data)

        return BatchResult(
            ok=True,
            input_message_id=input_message_id,
            status_code=status,
            response=data,
            error=None,
        )

    except Exception as e:
        return BatchResult(
            ok=False,
            input_message_id=input_message_id,
            status_code=None,
            response=None,
            error=str(e),
        )


async def run_batch(
    url: str,
    messages: List[Dict[str, Any]],
    out_path: Path,
    err_path: Path,
    concurrency: int,
    timeout_s: float,
    limit: Optional[int],
) -> None:
    """
    Runs the batch with bounded concurrency and writes:
      - out_path: JSONL of successful triage responses
      - err_path: JSONL of failures with reason
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional limit for quick smoke tests (e.g., first 5)
    if limit is not None:
        messages = messages[:limit]

    sem = asyncio.Semaphore(concurrency)

    major_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()

    async with httpx.AsyncClient() as client, \
            out_path.open("w", encoding="utf-8") as out_f, \
            err_path.open("w", encoding="utf-8") as err_f:

        async def worker(msg: Dict[str, Any]) -> None:
            async with sem:
                res = await post_one(client, url, msg, timeout_s)

            if res.ok and res.response:
                # Write the response (JSONL)
                out_f.write(json.dumps(res.response, ensure_ascii=False) + "\n")

                # Tally distributions
                major = res.response.get("major_category", "unknown")
                sub_key = res.response.get("sub_action_key", "unknown")
                major_counts[str(major)] += 1
                action_counts[str(sub_key)] += 1
            else:
                # Write a structured error record
                err_record = {
                    "input_message_id": res.input_message_id,
                    "status_code": res.status_code,
                    "error": res.error,
                }
                err_f.write(json.dumps(err_record, ensure_ascii=False) + "\n")

        await asyncio.gather(*(worker(m) for m in messages))

    # Print summary
    total = len(messages)
    ok_total = sum(major_counts.values())
    fail_total = total - ok_total

    print("\n===== Batch Summary =====")
    print(f"Total messages: {total}")
    print(f"Succeeded:     {ok_total}")
    print(f"Failed:        {fail_total}")
    print(f"Results file:  {out_path}")
    print(f"Errors file:   {err_path}")

    print("\nMajor category distribution:")
    for k, v in major_counts.most_common():
        print(f"  {k:28s} {v}")

    print("\nSub-action-key distribution:")
    for k, v in action_counts.most_common(20):
        print(f"  {k:28s} {v}")
    if len(action_counts) > 20:
        print(f"  ... ({len(action_counts) - 20} more)")


def main() -> None:
    p = argparse.ArgumentParser(description="Batch test Gmail triage endpoint with a JSON array fixture.")
    p.add_argument("--url", required=True, help="API endpoint URL, e.g. http://localhost:8000/v1/gmail/triage")
    p.add_argument("--input", required=True, help="Path to JSON array of Gmail messages")
    p.add_argument("--out", default="outputs/triage_results.jsonl", help="Where to write successful outputs (JSONL)")
    p.add_argument("--errors", default="outputs/triage_errors.jsonl", help="Where to write errors (JSONL)")
    p.add_argument("--concurrency", type=int, default=3, help="How many requests in flight at once")
    p.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout seconds")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N messages")

    args = p.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.out)
    err_path = Path(args.errors)

    messages = load_messages(input_path)
    asyncio.run(
        run_batch(
            url=args.url,
            messages=messages,
            out_path=out_path,
            err_path=err_path,
            concurrency=max(1, args.concurrency),
            timeout_s=args.timeout,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
