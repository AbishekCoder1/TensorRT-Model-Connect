from tools import ai_agent_system


def test_task_body_contains_required_sections() -> None:
    body = ai_agent_system.task_body(
        scope="docs only",
        change="remove stale text",
        acceptance=["stale text is gone"],
        verification=["rg stale docs || true"],
        non_goals=["no runtime edits"],
        risk="Low.",
    )

    assert ai_agent_system.validate_task_description(body) == []
    assert "## Scope" in body
    assert "- stale text is gone" in body


def test_validate_task_description_reports_missing_sections() -> None:
    missing = ai_agent_system.validate_task_description(
        "## Scope\nOne file.\n\n## Change\nDo it.\n"
    )

    assert missing == ["Acceptance Criteria", "Verification", "Non-goals"]


def test_encoded_project_path_escapes_namespace() -> None:
    assert ai_agent_system.encoded_project("yifeif/trt-transformers") == "yifeif%2Ftrt-transformers"
    assert ai_agent_system.encoded_project("12345") == "12345"


def test_is_ai_promotion_schedule_accepts_explicit_variable() -> None:
    schedule = {
        "description": "Periodic maintenance",
        "variables": [{"key": "AI_STAGING_PROMOTE", "value": "1"}],
    }

    assert ai_agent_system.is_ai_promotion_schedule(schedule)


def test_is_ai_promotion_schedule_accepts_descriptive_name() -> None:
    schedule = {"description": "AI staging promotion MR", "variables": []}

    assert ai_agent_system.is_ai_promotion_schedule(schedule)


def test_is_ai_promotion_schedule_rejects_unrelated_schedule() -> None:
    schedule = {
        "description": "Nightly",
        "variables": [{"key": "AI_STAGING_PROMOTE", "value": "0"}],
    }

    assert not ai_agent_system.is_ai_promotion_schedule(schedule)
