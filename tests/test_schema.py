from datetime import datetime, timezone

from app.gmail import parse_gmail_message
from app.models import (
    Entities,
    ExtractedSummary,
    GmailMessageInput,
    LLMTriageOutput,
    MajorCategory,
    RecommendedAction,
    UrgencySignals,
)
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


def test_postprocess_triage_fills_entities_and_debug():
    msg = GmailMessageInput.model_validate_json(
        open("scripts/example_gmail_input.json", "r", encoding="utf-8").read()
    )
    parsed = parse_gmail_message(msg)

    llm = LLMTriageOutput(
        major_category=MajorCategory.schedule_and_time,
        sub_action_key="SCHEDULE_CONFIRM_TIME",
        explicit_task=False,
        confidence=0.87,
        suggested_reply_action=["Yes, Friday at 2pm PT works for me."],
        task_proposal=None,
        recommended_actions=[
            RecommendedAction(key="reply_confirm", label="Confirm Time", kind="PRIMARY", rank=1),
        ],
        urgency_signals=UrgencySignals(
            urgency="high",
            deadline_detected=True,
            deadline_text="Friday 2pm PT",
            reply_by=None,
            reason="Sender is waiting on your confirmation for a meeting.",
        ),
        extracted_summary=ExtractedSummary(
            ask="Confirm whether Friday at 2pm PT works.",
            success_criteria="Reply confirming or proposing an alternative time.",
            missing_info=[],
        ),
        evidence=["Can you confirm Friday at 2pm PT works?"],
        entities=Entities.model_validate({
            "people": [],
            "dates": [{"text": "Friday 2pm PT", "iso": None, "type": "meeting_time"}],
            "money": [],
            "docs": [],
            "meeting": None,
        }),
    )

    resp = postprocess_triage(
        msg=msg,
        parsed_email=parsed,
        llm=llm,
        predicted_at=datetime.now(timezone.utc),
    )

    # Response is wrapped in { output: { ... } }
    output = resp.output

    assert output.major_category == MajorCategory.schedule_and_time
    assert output.sub_action_key == "SCHEDULE_CONFIRM_TIME"
    assert output.confidence >= 0.0 and output.confidence <= 1.0

    # Sender should be filled by postprocessor.
    assert any(p.role == "sender" and p.email == "sarah@acme.com" for p in output.entities.people)

    # Date iso should be filled if parsing succeeds.
    assert output.entities.dates[0].iso is not None

    # Meeting object should exist for schedule actions.
    assert output.entities.meeting is not None
    assert output.entities.meeting.topic is not None

    # Debug metadata should be injected.
    assert output.debug.analysis_timestamp != ""
    assert output.debug.model_version != ""
    assert output.debug.prompt_version != ""

    # Urgency signals should pass through.
    assert output.urgency_signals.urgency == "high"
    assert output.urgency_signals.deadline_detected is True

    # Suggested reply actions should pass through.
    assert len(output.suggested_reply_action) == 1

    # Recommended actions should pass through.
    assert len(output.recommended_actions) == 1
    assert output.recommended_actions[0].kind == "PRIMARY"

    # Extracted summary should pass through.
    assert output.extracted_summary.ask != ""
