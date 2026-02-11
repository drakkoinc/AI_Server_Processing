"""Small utility to call the AI server (local dev helper).

Conceptual purpose
------------------
When you're iterating on prompts, schemas, or postprocessing, you often want to quickly
send a sample payload to the FastAPI server and inspect the JSON.

This script is intentionally tiny so it can live in your repo and be used by anyone
without additional setup.

Technical behavior
------------------
- Reads a JSON file from disk
- POSTs it to the provided URL
- Prints the response JSON (pretty-printed) or raw text on parse failure

Usage examples
--------------
Triage (recommended):
  python scripts/call_api.py --url http://localhost:8000/rd/api/v1/ai/triage --input scripts/example_gmail_input.json
"""

from __future__ import annotations

import argparse
import json

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--input", required=True, help="Path to JSON request payload.")
    args = ap.parse_args()

    payload = json.loads(open(args.input, "r", encoding="utf-8").read())

    with httpx.Client(timeout=60.0) as client:
        r = client.post(args.url, json=payload)
        print("Status:", r.status_code)
        try:
            print(json.dumps(r.json(), indent=2))
        except Exception:
            print(r.text)


if __name__ == "__main__":
    main()
