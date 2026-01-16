from datetime import datetime, timezone

from app.gmail import parse_gmail_message
from app.models import GmailMessageInput, LLMTriageOutput, MajorCategory
from app.postprocess import postprocess_triage


def test_gmail_message_parses_and_decodes_bodies():
    msg = GmailMessageInput.model_validate_json(
        open("scripts/example_gmail_input.json", "r", encoding="utf-8").read()
    )
    parsed = parse_gmail_message(msg)

    assert parsed.message_id == "18c8a3f1b2d0a9a1"
    assert "Re:" in parsed.subject
    # The sample body is truncated base64 but should decode to something non-empty.
    assert parsed.body_text != ""


def test_postprocess_triage_injects_ids_and_fills_entities():
    msg = GmailMessageInput.model_validate_json(
        open("scripts/example_gmail_input.json", "r", encoding="utf-8").read()
    )
    parsed = parse_gmail_message(msg)

    llm = LLMTriageOutput(
        major_category=MajorCategory.schedule_and_time,
        sub_action_key="SCHEDULE_CONFIRM_TIME",
        reply_required=True,
        explicit_task=False,
        task_type=None,
        confidence=0.87,
        reason="Sender asks you to confirm the meeting time.",
        evidence=["Can you confirm Friday at 2pm PT works?"],
        entities={
            "people": [],
            "dates": [{"text": "Friday 2pm PT", "iso": None, "type": "meeting_time"}],
            "money": [],
            "docs": [],
            "meeting": None,
        },
    )

    resp = postprocess_triage(
        msg=msg,
        parsed_email=parsed,
        llm=llm,
        predicted_at=datetime.now(timezone.utc),
    )

    assert resp.message_id == msg.id
    assert resp.thread_id == msg.threadId
    assert resp.major_category == MajorCategory.schedule_and_time

    # Sender should be filled.
    assert any(p.role == "sender" and p.email == "sarah@acme.com" for p in resp.entities.people)

    # Date iso should be filled if parsing succeeds.
    assert resp.entities.dates[0].iso is not None

    # Meeting object should exist for schedule actions.
    assert resp.entities.meeting is not None
    assert resp.entities.meeting.topic is not None
