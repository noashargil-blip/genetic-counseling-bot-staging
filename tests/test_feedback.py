# -*- coding: utf-8 -*-
"""
Tests for the POST /feedback endpoint and app/feedback.py storage module.

Coverage
--------
* POST /feedback — happy paths and edge cases
* Privacy invariants: no question/answer text is stored
* JSONL log file creation and content format
* app/feedback.record() direct unit tests
* read_recent() helper
"""
import json
import pathlib
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import feedback as feedback_module

client = TestClient(app)


@pytest.fixture(autouse=True)
def _tmp_feedback_file(tmp_path, monkeypatch):
    """Redirect feedback output to a temporary file per test."""
    tmp_feedback = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback_module, "_FEEDBACK_DIR",  tmp_path)
    monkeypatch.setattr(feedback_module, "_FEEDBACK_FILE", tmp_feedback)
    return tmp_feedback


# ---------------------------------------------------------------------------
# POST /feedback — HTTP layer
# ---------------------------------------------------------------------------

class TestFeedbackEndpoint:
    def test_helpful_true_returns_200(self):
        resp = client.post("/feedback", json={"helpful": True})
        assert resp.status_code == 200

    def test_helpful_false_returns_200(self):
        resp = client.post("/feedback", json={"helpful": False})
        assert resp.status_code == 200

    def test_response_has_feedback_id(self):
        data = client.post("/feedback", json={"helpful": True}).json()
        assert "feedback_id" in data
        assert isinstance(data["feedback_id"], str)
        assert len(data["feedback_id"]) == 8  # uuid4 hex[:8]

    def test_response_has_recorded_true(self):
        data = client.post("/feedback", json={"helpful": True}).json()
        assert data["recorded"] is True

    def test_with_all_optional_fields(self):
        resp = client.post("/feedback", json={
            "helpful": False,
            "reason": "לא הבנתי את התשובה",
            "matched_topic": "vus",
            "safety_level": "general_information",
            "question_length": 42,
        })
        assert resp.status_code == 200
        assert resp.json()["recorded"] is True

    def test_reason_too_long_is_rejected(self):
        """FastAPI schema enforces max_length=200 on reason."""
        resp = client.post("/feedback", json={
            "helpful": True,
            "reason": "x" * 201,
        })
        assert resp.status_code == 422

    def test_missing_helpful_field_rejected(self):
        resp = client.post("/feedback", json={"reason": "test"})
        assert resp.status_code == 422

    def test_question_length_negative_rejected(self):
        resp = client.post("/feedback", json={"helpful": True, "question_length": -1})
        assert resp.status_code == 422

    def test_each_call_returns_unique_feedback_id(self):
        id1 = client.post("/feedback", json={"helpful": True}).json()["feedback_id"]
        id2 = client.post("/feedback", json={"helpful": True}).json()["feedback_id"]
        assert id1 != id2


# ---------------------------------------------------------------------------
# Privacy invariants
# ---------------------------------------------------------------------------

class TestFeedbackPrivacy:
    def test_no_question_text_in_request_schema(self):
        """The /feedback request schema must not have a 'question' field."""
        from app.main import FeedbackRequest  # noqa: PLC0415
        fields = FeedbackRequest.model_fields
        assert "question" not in fields, "'question' field must not be in FeedbackRequest"
        assert "answer" not in fields,   "'answer' field must not be in FeedbackRequest"

    def test_stored_entry_has_no_question_text(self, _tmp_feedback_file):
        client.post("/feedback", json={
            "helpful": True,
            "matched_topic": "carrier",
            "question_length": 55,
        })
        line = _tmp_feedback_file.read_text(encoding="utf-8").strip()
        entry = json.loads(line)
        assert "question" not in entry
        assert "answer" not in entry
        assert "text" not in entry

    def test_question_length_stored_not_text(self, _tmp_feedback_file):
        client.post("/feedback", json={
            "helpful": True,
            "question_length": 73,
        })
        entry = json.loads(_tmp_feedback_file.read_text(encoding="utf-8").strip())
        assert entry["question_length"] == 73


# ---------------------------------------------------------------------------
# JSONL storage format
# ---------------------------------------------------------------------------

class TestFeedbackStorage:
    def test_creates_log_file(self, _tmp_feedback_file):
        client.post("/feedback", json={"helpful": True})
        assert _tmp_feedback_file.exists()

    def test_log_file_is_valid_jsonl(self, _tmp_feedback_file):
        client.post("/feedback", json={"helpful": True})
        client.post("/feedback", json={"helpful": False, "reason": "unclear"})
        lines = [l for l in _tmp_feedback_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2
        for line in lines:
            entry = json.loads(line)  # must not raise
            assert isinstance(entry, dict)

    def test_entry_has_required_fields(self, _tmp_feedback_file):
        client.post("/feedback", json={"helpful": True, "matched_topic": "vus"})
        entry = json.loads(_tmp_feedback_file.read_text(encoding="utf-8").strip())
        for field in ("feedback_id", "timestamp_utc", "helpful", "matched_topic"):
            assert field in entry, f"missing field: {field}"

    def test_helpful_value_stored_correctly(self, _tmp_feedback_file):
        client.post("/feedback", json={"helpful": False})
        entry = json.loads(_tmp_feedback_file.read_text(encoding="utf-8").strip())
        assert entry["helpful"] is False

    def test_reason_stored_when_provided(self, _tmp_feedback_file):
        client.post("/feedback", json={"helpful": True, "reason": "מאוד עזר"})
        entry = json.loads(_tmp_feedback_file.read_text(encoding="utf-8").strip())
        assert entry["reason"] == "מאוד עזר"

    def test_reason_null_when_not_provided(self, _tmp_feedback_file):
        client.post("/feedback", json={"helpful": True})
        entry = json.loads(_tmp_feedback_file.read_text(encoding="utf-8").strip())
        assert entry["reason"] is None

    def test_timestamp_is_utc_iso(self, _tmp_feedback_file):
        client.post("/feedback", json={"helpful": True})
        entry = json.loads(_tmp_feedback_file.read_text(encoding="utf-8").strip())
        ts = entry["timestamp_utc"]
        assert "T" in ts and "Z" in ts, f"Timestamp not ISO-UTC: {ts}"

    def test_multiple_entries_appended(self, _tmp_feedback_file):
        for _ in range(5):
            client.post("/feedback", json={"helpful": True})
        lines = [l for l in _tmp_feedback_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 5


# ---------------------------------------------------------------------------
# app/feedback module — unit tests
# ---------------------------------------------------------------------------

class TestFeedbackModule:
    def test_record_returns_string_id(self):
        fid = feedback_module.record(helpful=True)
        assert isinstance(fid, str)
        assert len(fid) == 8

    def test_record_helpful_true(self, _tmp_feedback_file):
        feedback_module.record(helpful=True, matched_topic="carrier")
        entry = json.loads(_tmp_feedback_file.read_text(encoding="utf-8").strip())
        assert entry["helpful"] is True
        assert entry["matched_topic"] == "carrier"

    def test_record_reason_capped_at_200(self, _tmp_feedback_file):
        long_reason = "א" * 300
        feedback_module.record(helpful=False, reason=long_reason)
        entry = json.loads(_tmp_feedback_file.read_text(encoding="utf-8").strip())
        assert len(entry["reason"]) <= 200

    def test_read_recent_empty_when_no_file(self, _tmp_feedback_file):
        assert feedback_module.read_recent() == []

    def test_read_recent_returns_entries(self, _tmp_feedback_file):
        feedback_module.record(helpful=True)
        feedback_module.record(helpful=False, reason="test")
        entries = feedback_module.read_recent(n=10)
        assert len(entries) == 2
        assert all("feedback_id" in e for e in entries)

    def test_read_recent_respects_n_limit(self, _tmp_feedback_file):
        for _ in range(10):
            feedback_module.record(helpful=True)
        entries = feedback_module.read_recent(n=3)
        assert len(entries) == 3

    def test_preset_reasons_is_list(self):
        assert isinstance(feedback_module.PRESET_REASONS, list)
        assert len(feedback_module.PRESET_REASONS) >= 4

    def test_record_does_not_raise_on_write_error(self, tmp_path, monkeypatch):
        """Even if the log directory is not writable, record() must not raise."""
        # Point to an impossible path
        monkeypatch.setattr(feedback_module, "_FEEDBACK_DIR",  pathlib.Path("/nonexistent/path"))
        monkeypatch.setattr(feedback_module, "_FEEDBACK_FILE", pathlib.Path("/nonexistent/path/f.jsonl"))
        fid = feedback_module.record(helpful=True)  # must not raise
        assert isinstance(fid, str)
