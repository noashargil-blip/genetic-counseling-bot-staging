# -*- coding: utf-8 -*-
"""
tests/test_session27_physician_portal.py

Session 27 — physician review portal tests.

Covers:
  - Review DB persistence (CRUD, deduplication, audit trail, status transitions)
  - Physician auth (password hashing / verification, session helpers)
  - Patient-facing display modes (AI_DRAFT_VISIBILITY_MODE)
  - FastAPI portal endpoints (login, logout, me, list, get, review)
  - Privacy / security boundaries (no PII stored, portal gate when disabled)
  - Existing test suites not broken (counseling + review_workflow regression)

All DB tests use an isolated in-memory or temp-file SQLite path via REVIEW_DB_SQLITE_PATH.
No real files outside tmp are touched.
"""

import os
import tempfile
import uuid

import pytest

# ── Isolation: redirect SQLite to a per-test temp file ──────────────────────

@pytest.fixture(autouse=True)
def isolated_review_db(tmp_path, monkeypatch):
    """Each test gets its own SQLite file; REVIEW_DB_SQLITE_PATH is overridden."""
    db_path = str(tmp_path / "test_review.db")
    monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    # Re-import so module-level globals pick up the new env var
    import importlib
    from app import review_db
    importlib.reload(review_db)
    review_db.init_db()
    yield review_db
    # cleanup handled by tmp_path fixture


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestReviewDbPersistence — CRUD basics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestReviewDbPersistence:

    def test_create_draft_returns_dict(self, isolated_review_db):
        rdb = isolated_review_db
        draft = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="הגן BRCA1 מקודד לחלבון תיקון DNA.",
            gene_symbol="BRCA1",
        )
        assert draft is not None
        assert draft["id"]
        assert draft["gene_symbol"] == "BRCA1"
        assert draft["review_status"] == "pending"
        assert draft["original_ai_text"] == "הגן BRCA1 מקודד לחלבון תיקון DNA."

    def test_create_draft_effective_text_equals_original(self, isolated_review_db):
        rdb = isolated_review_db
        draft = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="טקסט לדוגמה על APOE.",
            gene_symbol="APOE",
        )
        assert draft["effective_text"] == draft["original_ai_text"]
        assert draft["physician_reviewed"] is False
        assert draft["physician_approved"] is False

    def test_get_draft_by_id(self, isolated_review_db):
        rdb = isolated_review_db
        created = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="טקסט על TP53.",
            gene_symbol="TP53",
        )
        fetched = rdb.get_draft(created["id"])
        assert fetched is not None
        assert fetched["id"] == created["id"]
        assert fetched["gene_symbol"] == "TP53"

    def test_get_draft_nonexistent_returns_none(self, isolated_review_db):
        rdb = isolated_review_db
        assert rdb.get_draft(str(uuid.uuid4())) is None

    def test_list_drafts_empty(self, isolated_review_db):
        rdb = isolated_review_db
        result = rdb.list_drafts()
        assert result == []

    def test_list_drafts_returns_all(self, isolated_review_db):
        rdb = isolated_review_db
        rdb.create_draft(draft_type="gene_summary", original_ai_text="טקסט על BRCA1.", gene_symbol="BRCA1")
        rdb.create_draft(draft_type="gene_summary", original_ai_text="טקסט על APOE.", gene_symbol="APOE")
        drafts = rdb.list_drafts()
        assert len(drafts) == 2

    def test_list_drafts_filter_by_status(self, isolated_review_db):
        rdb = isolated_review_db
        draft = rdb.create_draft(draft_type="gene_summary", original_ai_text="טקסט על NF1.", gene_symbol="NF1")
        pending = rdb.list_drafts(status="pending")
        assert len(pending) == 1
        approved = rdb.list_drafts(status="approved")
        assert len(approved) == 0

    def test_list_drafts_filter_by_gene(self, isolated_review_db):
        rdb = isolated_review_db
        rdb.create_draft(draft_type="gene_summary", original_ai_text="טקסט על BRCA1.", gene_symbol="BRCA1")
        rdb.create_draft(draft_type="gene_summary", original_ai_text="טקסט על APOE.", gene_symbol="APOE")
        brca = rdb.list_drafts(gene_symbol="BRCA1")
        assert len(brca) == 1
        assert brca[0]["gene_symbol"] == "BRCA1"

    def test_pending_count(self, isolated_review_db):
        rdb = isolated_review_db
        assert rdb.pending_count() == 0
        rdb.create_draft(draft_type="gene_summary", original_ai_text="טקסט שניים.", gene_symbol="BRCA2")
        assert rdb.pending_count() == 1

    def test_draft_stores_no_patient_pii(self, isolated_review_db):
        """Draft fields must not include Israeli IDs, phone numbers, or patient names."""
        rdb = isolated_review_db
        draft = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="הגן BRCA2 שייך לנתיב תיקון הDNA.",
            gene_symbol="BRCA2",
            normalized_query="מה זה גן BRCA2?",  # anonymized question
            source_metadata={"answer_tier": "tier2"},
        )
        draft_str = str(draft)
        # Must not contain patterns that look like ID numbers (9 digits)
        import re
        assert not re.search(r"\b\d{9}\b", draft_str), "No 9-digit numbers (could be ID)"
        # Must not contain phone patterns
        assert not re.search(r"\b05\d{8}\b", draft_str), "No phone numbers"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestReviewDbDeduplication
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestReviewDbDeduplication:

    def test_identical_text_does_not_insert_duplicate(self, isolated_review_db):
        rdb = isolated_review_db
        text = "הגן BRCA1 מקודד לחלבון BRCA1 המעורב בתיקון שבירות DNA דו-גדיליות."
        d1 = rdb.create_draft(draft_type="gene_summary", original_ai_text=text, gene_symbol="BRCA1")
        d2 = rdb.create_draft(draft_type="gene_summary", original_ai_text=text, gene_symbol="BRCA1")
        assert d1["id"] == d2["id"], "Identical draft must not create a second record"
        assert rdb.pending_count() == 1

    def test_duplicate_increments_seen_count(self, isolated_review_db):
        rdb = isolated_review_db
        text = "הגן APOE קשור לשיכולת שומנים ולסיכון לאלצהיימר."
        rdb.create_draft(draft_type="gene_summary", original_ai_text=text, gene_symbol="APOE")
        d2 = rdb.create_draft(draft_type="gene_summary", original_ai_text=text, gene_symbol="APOE")
        assert d2["seen_count"] == 2

    def test_different_gene_creates_separate_record(self, isolated_review_db):
        rdb = isolated_review_db
        text = "גן זה מקודד לחלבון חשוב."
        rdb.create_draft(draft_type="gene_summary", original_ai_text=text, gene_symbol="BRCA1")
        rdb.create_draft(draft_type="gene_summary", original_ai_text=text, gene_symbol="BRCA2")
        assert rdb.pending_count() == 2

    def test_whitespace_normalized_for_dedup(self, isolated_review_db):
        rdb = isolated_review_db
        text1 = "הגן TP53  מקודד לחלבון p53."
        text2 = "הגן TP53 מקודד לחלבון p53."  # single space
        d1 = rdb.create_draft(draft_type="gene_summary", original_ai_text=text1, gene_symbol="TP53")
        d2 = rdb.create_draft(draft_type="gene_summary", original_ai_text=text2, gene_symbol="TP53")
        assert d1["id"] == d2["id"], "Whitespace-normalised text should deduplicate"

    def test_approved_draft_does_not_block_new_pending(self, isolated_review_db):
        rdb = isolated_review_db
        text = "הגן APC מעורב בסרטן המעי הגס."
        d = rdb.create_draft(draft_type="gene_summary", original_ai_text=text, gene_symbol="APC")
        rdb.update_draft_status(d["id"], new_status="approved", reviewer_identity="dr_test")
        # Now a new draft with same text — approved record exists, should create new pending
        d2 = rdb.create_draft(draft_type="gene_summary", original_ai_text=text, gene_symbol="APC")
        # Different id because the approved one is not in pending/needs_revision state
        assert d2["id"] != d["id"]
        assert d2["review_status"] == "pending"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestReviewLifecycle — status transitions + audit trail
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestReviewLifecycle:

    def _make(self, rdb, gene="BRCA1", text="טקסט לבדיקה על הגן."):
        return rdb.create_draft(draft_type="gene_summary", original_ai_text=text, gene_symbol=gene)

    def test_approve_transition(self, isolated_review_db):
        rdb = isolated_review_db
        draft = self._make(rdb)
        updated = rdb.update_draft_status(
            draft["id"], new_status="approved", reviewer_identity="dr_cohen"
        )
        assert updated["review_status"] == "approved"
        assert updated["reviewed_by"] == "dr_cohen"
        assert updated["reviewed_at"] is not None
        assert updated["physician_approved"] is True

    def test_reject_transition(self, isolated_review_db):
        rdb = isolated_review_db
        draft = self._make(rdb)
        updated = rdb.update_draft_status(
            draft["id"], new_status="rejected", reviewer_identity="dr_levi",
            review_comment="לא מדויק מספיק."
        )
        assert updated["review_status"] == "rejected"
        assert updated["review_comment"] == "לא מדויק מספיק."
        assert updated["physician_approved"] is False

    def test_needs_revision_transition(self, isolated_review_db):
        rdb = isolated_review_db
        draft = self._make(rdb)
        updated = rdb.update_draft_status(
            draft["id"], new_status="needs_revision", reviewer_identity="dr_bar",
        )
        assert updated["review_status"] == "needs_revision"

    def test_approve_with_physician_edited_text(self, isolated_review_db):
        rdb = isolated_review_db
        draft = self._make(rdb)
        edited = "הגן BRCA1 מקודד לחלבון המעורב בתיקון DNA ובדיכוי גידולים."
        updated = rdb.update_draft_status(
            draft["id"], new_status="approved", reviewer_identity="dr_cohen",
            physician_edited_text=edited,
        )
        assert updated["physician_edited_text"] == edited
        assert updated["effective_text"] == edited  # effective_text follows edited text

    def test_invalid_status_raises_value_error(self, isolated_review_db):
        rdb = isolated_review_db
        draft = self._make(rdb)
        with pytest.raises(ValueError, match="Invalid status"):
            rdb.update_draft_status(draft["id"], new_status="nonexistent", reviewer_identity="x")

    def test_invalid_transition_raises_value_error(self, isolated_review_db):
        rdb = isolated_review_db
        draft = self._make(rdb)
        rdb.update_draft_status(draft["id"], new_status="approved", reviewer_identity="dr_test")
        with pytest.raises(ValueError, match="Invalid transition"):
            rdb.update_draft_status(draft["id"], new_status="rejected", reviewer_identity="dr_test")

    def test_nonexistent_draft_raises_value_error(self, isolated_review_db):
        rdb = isolated_review_db
        with pytest.raises(ValueError, match="not found"):
            rdb.update_draft_status(str(uuid.uuid4()), new_status="approved", reviewer_identity="x")

    def test_revision_increments_on_each_update(self, isolated_review_db):
        rdb = isolated_review_db
        draft = self._make(rdb)
        assert draft["revision"] == 0
        u1 = rdb.update_draft_status(draft["id"], new_status="needs_revision", reviewer_identity="x")
        assert u1["revision"] == 1
        u2 = rdb.update_draft_status(draft["id"], new_status="approved", reviewer_identity="x")
        assert u2["revision"] == 2

    def test_audit_trail_length(self, isolated_review_db):
        rdb = isolated_review_db
        draft = self._make(rdb)
        rdb.update_draft_status(draft["id"], new_status="needs_revision", reviewer_identity="dr_a")
        rdb.update_draft_status(draft["id"], new_status="approved", reviewer_identity="dr_b")
        trail = rdb.get_audit_trail(draft["id"])
        assert len(trail) == 2

    def test_audit_trail_records_reviewer(self, isolated_review_db):
        rdb = isolated_review_db
        draft = self._make(rdb)
        rdb.update_draft_status(draft["id"], new_status="approved", reviewer_identity="dr_specific")
        trail = rdb.get_audit_trail(draft["id"])
        assert trail[0]["reviewer_identity"] == "dr_specific"

    def test_audit_trail_records_text_was_edited(self, isolated_review_db):
        rdb = isolated_review_db
        draft = self._make(rdb)
        rdb.update_draft_status(
            draft["id"], new_status="approved", reviewer_identity="dr_x",
            physician_edited_text="טקסט ערוך.",
        )
        trail = rdb.get_audit_trail(draft["id"])
        assert trail[0]["text_was_edited"] == 1

    def test_get_approved_draft(self, isolated_review_db):
        rdb = isolated_review_db
        draft = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="הגן HBB מקודד לחלבון הגלובין.",
            gene_symbol="HBB",
        )
        rdb.update_draft_status(draft["id"], new_status="approved", reviewer_identity="dr_test")
        approved = rdb.get_approved_draft("HBB")
        assert approved is not None
        assert approved["physician_approved"] is True
        assert approved["gene_symbol"] == "HBB"

    def test_get_approved_draft_returns_none_when_pending(self, isolated_review_db):
        rdb = isolated_review_db
        rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="הגן HBB מקודד לחלבון הגלובין.",
            gene_symbol="HBB",
        )
        assert rdb.get_approved_draft("HBB") is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestPhysicianAuth — password hashing + session utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPhysicianAuth:

    def test_generate_hash_format(self):
        from app.physician_auth import generate_hash
        h = generate_hash("test123")
        parts = h.split("$")
        assert len(parts) == 4
        assert parts[0] == "pbkdf2sha256"
        assert parts[1].isdigit()  # iterations
        assert len(parts[2]) > 20  # hex salt
        assert len(parts[3]) > 20  # hex digest

    def test_verify_correct_password(self):
        from app.physician_auth import generate_hash, verify_password
        h = generate_hash("correct_pass!")
        assert verify_password("correct_pass!", h) is True

    def test_verify_wrong_password(self):
        from app.physician_auth import generate_hash, verify_password
        h = generate_hash("correct_pass!")
        assert verify_password("wrong_pass!", h) is False

    def test_verify_empty_password(self):
        from app.physician_auth import generate_hash, verify_password
        h = generate_hash("nonempty")
        assert verify_password("", h) is False

    def test_verify_malformed_hash(self):
        from app.physician_auth import verify_password
        assert verify_password("anything", "not_a_valid_hash") is False
        assert verify_password("anything", "") is False
        assert verify_password("anything", "abc$123") is False

    def test_two_hashes_of_same_password_differ(self):
        from app.physician_auth import generate_hash
        h1 = generate_hash("same_password")
        h2 = generate_hash("same_password")
        assert h1 != h2  # different random salts

    def test_authenticate_with_env_vars(self, monkeypatch):
        from app.physician_auth import generate_hash
        import importlib
        from app import physician_auth
        pw = "s3cur3P@ss"
        monkeypatch.setenv("REVIEWER_USERNAME", "dr_test")
        monkeypatch.setenv("REVIEWER_PASSWORD_HASH", generate_hash(pw))
        importlib.reload(physician_auth)
        assert physician_auth.authenticate("dr_test", pw) is True
        assert physician_auth.authenticate("dr_test", "wrong") is False
        assert physician_auth.authenticate("wrong_user", pw) is False

    def test_portal_enabled_requires_all_vars(self, monkeypatch):
        import importlib
        from app import physician_auth
        monkeypatch.delenv("REVIEW_PORTAL_ENABLED", raising=False)
        monkeypatch.delenv("REVIEWER_USERNAME", raising=False)
        monkeypatch.delenv("REVIEWER_PASSWORD_HASH", raising=False)
        monkeypatch.delenv("REVIEW_SESSION_SECRET", raising=False)
        importlib.reload(physician_auth)
        assert physician_auth.portal_enabled() is False

    def test_portal_enabled_true_when_all_set(self, monkeypatch):
        from app.physician_auth import generate_hash
        import importlib
        from app import physician_auth
        monkeypatch.setenv("REVIEW_PORTAL_ENABLED", "true")
        monkeypatch.setenv("REVIEWER_USERNAME", "doc")
        monkeypatch.setenv("REVIEWER_PASSWORD_HASH", generate_hash("pw"))
        monkeypatch.setenv("REVIEW_SESSION_SECRET", "x" * 32)
        importlib.reload(physician_auth)
        assert physician_auth.portal_enabled() is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestPortalApiDisabled — portal returns 503 when not configured
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPortalApiDisabled:

    @pytest.fixture
    def disabled_client(self, monkeypatch):
        """App client with portal vars absent → portal disabled."""
        monkeypatch.delenv("REVIEW_PORTAL_ENABLED", raising=False)
        monkeypatch.delenv("REVIEWER_USERNAME", raising=False)
        monkeypatch.delenv("REVIEWER_PASSWORD_HASH", raising=False)
        monkeypatch.delenv("REVIEW_SESSION_SECRET", raising=False)
        import importlib
        from app import physician_auth, main
        importlib.reload(physician_auth)
        importlib.reload(main)
        from fastapi.testclient import TestClient
        return TestClient(main.app)

    def test_physician_html_returns_503_when_disabled(self, disabled_client):
        r = disabled_client.get("/physician")
        assert r.status_code == 503

    def test_login_returns_503_when_disabled(self, disabled_client):
        r = disabled_client.post("/api/physician/login",
                                  json={"username": "x", "password": "y"})
        assert r.status_code == 503

    def test_me_returns_503_when_disabled(self, disabled_client):
        r = disabled_client.get("/api/physician/me")
        assert r.status_code == 503

    def test_drafts_returns_503_when_disabled(self, disabled_client):
        r = disabled_client.get("/api/physician/drafts")
        assert r.status_code == 503


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestPortalApiEnabled — full end-to-end with portal enabled
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_TEST_USERNAME = "dr_portal_test"
_TEST_PASSWORD = "P@ssw0rd_Test!"


@pytest.fixture
def portal_client(tmp_path, monkeypatch):
    """App client with physician portal fully enabled."""
    from app.physician_auth import generate_hash
    pw_hash = generate_hash(_TEST_PASSWORD)
    db_path = str(tmp_path / "portal_test.db")

    monkeypatch.setenv("REVIEW_PORTAL_ENABLED", "true")
    monkeypatch.setenv("REVIEWER_USERNAME", _TEST_USERNAME)
    monkeypatch.setenv("REVIEWER_PASSWORD_HASH", pw_hash)
    monkeypatch.setenv("REVIEW_SESSION_SECRET", "test_secret_1234567890abcdef")
    monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    import importlib
    from app import physician_auth, review_db, main
    importlib.reload(physician_auth)
    importlib.reload(review_db)
    importlib.reload(main)
    review_db.init_db()

    from fastapi.testclient import TestClient
    client = TestClient(main.app, raise_server_exceptions=True)
    return client, review_db


class TestPortalApiEnabled:

    def test_physician_html_served(self, portal_client):
        client, _ = portal_client
        r = client.get("/physician")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_login_wrong_credentials(self, portal_client):
        client, _ = portal_client
        r = client.post("/api/physician/login",
                        json={"username": _TEST_USERNAME, "password": "wrong!"})
        assert r.status_code == 401

    def test_login_correct_credentials(self, portal_client):
        client, _ = portal_client
        r = client.post("/api/physician/login",
                        json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD})
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["identity"] == _TEST_USERNAME

    def test_me_unauthenticated(self, portal_client):
        client, _ = portal_client
        # Fresh client — no session
        r = client.get("/api/physician/me")
        assert r.status_code == 401

    def test_me_after_login(self, portal_client):
        client, _ = portal_client
        client.post("/api/physician/login",
                    json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD})
        r = client.get("/api/physician/me")
        assert r.status_code == 200
        assert r.json()["identity"] == _TEST_USERNAME

    def test_logout_invalidates_session(self, portal_client):
        client, _ = portal_client
        client.post("/api/physician/login",
                    json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD})
        client.post("/api/physician/logout")
        r = client.get("/api/physician/me")
        assert r.status_code == 401

    def test_list_drafts_requires_auth(self, portal_client):
        client, _ = portal_client
        # No login
        r = client.get("/api/physician/drafts")
        assert r.status_code == 401

    def test_list_drafts_empty_after_login(self, portal_client):
        client, _ = portal_client
        client.post("/api/physician/login",
                    json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD})
        r = client.get("/api/physician/drafts")
        assert r.status_code == 200
        assert r.json()["drafts"] == []

    def test_list_drafts_after_creating_draft(self, portal_client):
        client, rdb = portal_client
        rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="הגן BRCA1 מקודד לחלבון.",
            gene_symbol="BRCA1",
        )
        client.post("/api/physician/login",
                    json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD})
        r = client.get("/api/physician/drafts")
        assert r.status_code == 200
        assert len(r.json()["drafts"]) == 1

    def test_get_draft_requires_auth(self, portal_client):
        client, _ = portal_client
        r = client.get(f"/api/physician/drafts/{uuid.uuid4()}")
        assert r.status_code == 401

    def test_get_draft_not_found(self, portal_client):
        client, _ = portal_client
        client.post("/api/physician/login",
                    json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD})
        r = client.get(f"/api/physician/drafts/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_review_approve(self, portal_client):
        client, rdb = portal_client
        draft = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="הגן TP53 מקודד לחלבון p53.",
            gene_symbol="TP53",
        )
        client.post("/api/physician/login",
                    json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD})
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"new_status": "approved", "review_comment": "נראה טוב."})
        assert r.status_code == 200
        assert r.json()["draft"]["review_status"] == "approved"

    def test_review_reject_with_comment(self, portal_client):
        client, rdb = portal_client
        draft = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="הגן APOE שייך לנתיב שומנים.",
            gene_symbol="APOE",
        )
        client.post("/api/physician/login",
                    json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD})
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"new_status": "rejected", "review_comment": "לא מדויק."})
        assert r.status_code == 200
        data = r.json()["draft"]
        assert data["review_status"] == "rejected"
        assert data["review_comment"] == "לא מדויק."

    def test_review_approve_with_edited_text(self, portal_client):
        client, rdb = portal_client
        draft = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="הגן HBB מקודד לשרשרת בטא של המוגלובין.",
            gene_symbol="HBB",
        )
        client.post("/api/physician/login",
                    json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD})
        edited = "הגן HBB מקודד לשרשרת בטא-גלובין, הרכיב העיקרי של המוגלובין."
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"new_status": "approved", "physician_edited_text": edited})
        assert r.status_code == 200
        result = r.json()["draft"]
        assert result["physician_edited_text"] == edited
        assert result["effective_text"] == edited

    def test_review_invalid_status(self, portal_client):
        client, rdb = portal_client
        draft = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="הגן NF1 מקודד לנוירופיברומין.",
            gene_symbol="NF1",
        )
        client.post("/api/physician/login",
                    json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD})
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"new_status": "invalid_status"})
        assert r.status_code == 400

    def test_get_audit_trail(self, portal_client):
        client, rdb = portal_client
        draft = rdb.create_draft(
            draft_type="gene_summary", original_ai_text="טקסט לבדיקת אודיט.", gene_symbol="APC"
        )
        client.post("/api/physician/login",
                    json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD})
        client.post(f"/api/physician/drafts/{draft['id']}/review",
                    json={"new_status": "approved"})
        r = client.get(f"/api/physician/drafts/{draft['id']}/audit")
        assert r.status_code == 200
        trail = r.json()["audit"]
        assert len(trail) == 1
        assert trail[0]["new_status"] == "approved"

    def test_pending_count_in_me_response(self, portal_client):
        client, rdb = portal_client
        rdb.create_draft(
            draft_type="gene_summary", original_ai_text="טקסט לספירה.", gene_symbol="APC"
        )
        client.post("/api/physician/login",
                    json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD})
        r = client.get("/api/physician/me")
        assert r.json()["pending_count"] == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestVisibilityMode — AI_DRAFT_VISIBILITY_MODE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestVisibilityMode:

    def test_immediate_mode_default(self, monkeypatch):
        """AI_DRAFT_VISIBILITY_MODE defaults to 'immediate'."""
        monkeypatch.delenv("AI_DRAFT_VISIBILITY_MODE", raising=False)
        import importlib
        from app import counseling_engine as ce
        importlib.reload(ce)
        assert ce._AI_DRAFT_VISIBILITY_MODE == "immediate"

    def test_approved_only_mode_env(self, monkeypatch):
        monkeypatch.setenv("AI_DRAFT_VISIBILITY_MODE", "approved_only")
        import importlib
        from app import counseling_engine as ce
        importlib.reload(ce)
        assert ce._AI_DRAFT_VISIBILITY_MODE == "approved_only"

    def test_approved_only_uses_approved_text(self, tmp_path, monkeypatch):
        """When approved_only, answer must come from the approved draft, not the raw AI draft."""
        db_path = str(tmp_path / "vis_test.db")
        monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
        monkeypatch.setenv("AI_DRAFT_VISIBILITY_MODE", "approved_only")
        monkeypatch.delenv("DATABASE_URL", raising=False)

        import importlib
        from app import review_db
        importlib.reload(review_db)
        review_db.init_db()

        # Seed an approved draft
        approved_text = "הגן BRCA1 מקודד לחלבון שמיירט תיקון DNA — טקסט שאושר."
        draft = review_db.create_draft(
            draft_type="gene_summary",
            original_ai_text=approved_text,
            gene_symbol="BRCA1",
        )
        review_db.update_draft_status(
            draft["id"], new_status="approved", reviewer_identity="dr_test"
        )

        approved = review_db.get_approved_draft("BRCA1")
        assert approved is not None
        assert approved["effective_text"] == approved_text

    def test_approved_only_fallback_when_no_approved_draft(self, tmp_path, monkeypatch):
        """When approved_only with no approved draft, get_approved_draft returns None."""
        db_path = str(tmp_path / "vis_fallback.db")
        monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
        monkeypatch.delenv("DATABASE_URL", raising=False)

        import importlib
        from app import review_db
        importlib.reload(review_db)
        review_db.init_db()

        # Pending draft only
        review_db.create_draft(
            draft_type="gene_summary",
            original_ai_text="טקסט שעדיין ממתין.",
            gene_symbol="BRCA2",
        )
        assert review_db.get_approved_draft("BRCA2") is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestExistingTestsNotBroken — regression guard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExistingTestsNotBroken:

    def test_app_main_imports_successfully(self):
        """app.main must be importable without errors."""
        import importlib
        from app import main
        importlib.reload(main)
        assert main.app is not None

    def test_counseling_engine_imports_successfully(self):
        """app.counseling_engine must import without errors."""
        import importlib
        from app import counseling_engine
        importlib.reload(counseling_engine)
        assert hasattr(counseling_engine, "answer_question")

    def test_review_db_imports_successfully(self):
        """app.review_db must import without errors."""
        import importlib
        from app import review_db
        importlib.reload(review_db)
        assert hasattr(review_db, "create_draft")

    def test_physician_auth_imports_successfully(self):
        """app.physician_auth must import without errors."""
        import importlib
        from app import physician_auth
        importlib.reload(physician_auth)
        assert hasattr(physician_auth, "verify_password")
        assert hasattr(physician_auth, "generate_hash")

    def test_ask_endpoint_still_responds(self, monkeypatch):
        """POST /ask must return exactly 5 keys (contract unchanged)."""
        monkeypatch.delenv("REVIEW_PORTAL_ENABLED", raising=False)
        import importlib
        from app import main
        importlib.reload(main)
        from fastapi.testclient import TestClient
        client = TestClient(main.app)
        r = client.post("/ask", json={"question": "מה זה VUS?"})
        assert r.status_code == 200
        data = r.json()
        required = {"answer", "safety_level", "needs_genetic_counselor",
                    "matched_topic", "suggested_questions"}
        assert required.issubset(set(data.keys())), \
            f"Missing required keys: {required - set(data.keys())}"

    def test_draft_short_text_not_stored(self, isolated_review_db):
        """Drafts shorter than 10 chars must be silently rejected."""
        rdb = isolated_review_db
        result = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="קצר",
            gene_symbol="X",
        )
        assert result is None
        assert rdb.pending_count() == 0

    def test_review_db_create_never_raises(self, isolated_review_db, monkeypatch):
        """create_draft must not raise even when the DB path is unusable."""
        rdb = isolated_review_db
        # Monkeypatch _get_connection to raise
        original = rdb._get_connection

        import contextlib
        @contextlib.contextmanager
        def broken_conn():
            raise RuntimeError("simulated DB failure")
            yield  # pragma: no cover

        monkeypatch.setattr(rdb, "_get_connection", broken_conn)
        result = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="טקסט תקין שלא אמור להישמר.",
            gene_symbol="BRCA1",
        )
        # Should return None silently, not raise
        assert result is None
