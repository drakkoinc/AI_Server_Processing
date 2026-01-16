"""
stream_one_email.py

Interactive one-email-at-a-time tester for the Drakko Gmail triage API.
Each run creates its own subfolder under scripts/runs/
scripts/runs/triage_01_02_26/
    ├── triage_run.jsonl
    ├── 001_<message_id>_output.json
    ├── 002_<message_id>_output.json
"""
import json
import re
from pathlib import Path
from datetime import datetime
import httpx
API_URL = "http://localhost:8000/v1/gmail/triage"
INPUT_FILE = Path(__file__).parent / "gmail_messages_sample_50.json"

# Base runs directory
RUNS_ROOT = Path(__file__).parent / "runs"
RUNS_ROOT.mkdir(exist_ok=True)

def create_run_folder():
    """
    Create a unique folder for this triage run.

    Base name format:
        triage_MM_DD_YY

    If that folder already exists, auto-increment:
        triage_MM_DD_YY_2
        triage_MM_DD_YY_3
    """
    date_tag = datetime.now().strftime("%m_%d_%y")
    base_name = f"triage_{date_tag}"

    run_dir = RUNS_ROOT / base_name
    counter = 2

    while run_dir.exists():
        run_dir = RUNS_ROOT / f"{base_name}_{counter}"
        counter += 1

    run_dir.mkdir()
    return run_dir

# Create the run-specific folder
RUN_DIR = create_run_folder()

# JSONL file lives inside the run folder
JSONL_LOG = RUN_DIR / "triage_run.jsonl"

def load_messages(path: Path):
    """
    Load Gmail messages from JSON array or JSONL file.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]

def get_header(msg: dict, name: str):
    """
    Extract a Gmail header value by name (case-insensitive).
    """
    headers = msg.get("payload", {}).get("headers", []) or []
    target = name.lower()
    for h in headers:
        if (h.get("name") or "").lower() == target:
            return h.get("value")
    return None

def safe_snippet(msg: dict, n: int = 180):
    """
    Normalize and truncate Gmail snippet text.
    """
    s = msg.get("snippet") or ""
    s = re.sub(r"\s+", " ", s).strip()
    return s[:n] + ("…" if len(s) > n else "")

def append_jsonl(path: Path, obj: dict):
    """
    Append a single JSON object to a JSONL file.
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# -----------------------------
# MAIN PROGRAM
# -----------------------------
def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Expected input file at: {INPUT_FILE}"
        )

    messages = load_messages(INPUT_FILE)

    print(f"Loaded {len(messages)} messages")
    print(f"Run directory: {RUN_DIR}")
    print(f"Run log file:  {JSONL_LOG}\n")

    client = httpx.Client(timeout=120.0)

    idx = 0
    while idx < len(messages):
        msg = messages[idx]

        message_id = msg.get("id") or msg.get("message_id") or f"index_{idx}"
        thread_id = msg.get("threadId") or msg.get("thread_id")

        subject = get_header(msg, "Subject") or "(no subject)"
        from_ = get_header(msg, "From") or "(no from)"
        to_ = get_header(msg, "To") or "(no to)"
        date_ = get_header(msg, "Date") or "(no date)"
        snippet = safe_snippet(msg)

        input_reference = {
            "index": idx + 1,
            "total": len(messages),
            "message_id": message_id,
            "thread_id": thread_id,
            "from": from_,
            "to": to_,
            "date": date_,
            "subject": subject,
            "snippet": snippet,
        }

        print("=" * 90)
        print(f"[{idx+1}/{len(messages)}] INPUT REFERENCE")
        print(f"message_id: {message_id}")
        if thread_id:
            print(f"thread_id : {thread_id}")
        print(f"from      : {from_}")
        print(f"to        : {to_}")
        print(f"date      : {date_}")
        print(f"subject   : {subject}")
        print(f"snippet   : {snippet}")

        cmd = input("\nPress Enter to TRIAGE | (n) next | (p) prev | (q) quit: ").strip().lower()

        if cmd == "q":
            break
        if cmd == "p":
            idx = max(0, idx - 1)
            continue
        if cmd == "n":
            idx += 1
            continue

        try:
            r = client.post(API_URL, json=msg)
            print(f"\nSTATUS: {r.status_code}")

            record = {
                "timestamp": datetime.now().isoformat(),
                "input_reference": input_reference,
                "http_status": r.status_code,
            }

            if r.status_code != 200:
                record["error_text"] = r.text
                append_jsonl(JSONL_LOG, record)
                print("ERROR RESPONSE:")
                print(r.text)

            else:
                out = r.json()
                record["output"] = out
                append_jsonl(JSONL_LOG, record)

                safe_mid = str(message_id).replace("/", "_")
                out_file = RUN_DIR / f"{idx+1:03d}_{safe_mid}_output.json"

                out_file.write_text(
                    json.dumps(
                        {"input_reference": input_reference, "output": out},
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                print("\nOUTPUT (key fields)")
                for k in ["major_category", "sub_action_key", "urgency", "confidence"]:
                    if k in out:
                        print(f"{k:14}: {out.get(k)}")

                print(f"\nSaved snapshot → {out_file}")

        except Exception as e:
            append_jsonl(
                JSONL_LOG,
                {
                    "timestamp": datetime.now().isoformat(),
                    "input_reference": input_reference,
                    "exception": repr(e),
                },
            )
            print("EXCEPTION:", e)

        idx += 1

    print("\nDone.")
    print(f"All outputs saved under:\n{RUN_DIR}")

if __name__ == "__main__":
    main()
