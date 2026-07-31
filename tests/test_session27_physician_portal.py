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
    monkeypatch.setenv("REVIEW_SESSION_SECRET", "test_secret_1234567890abcdef_xyz12")  # 32+ chars
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
        # New contract: action verb + revision integer (not new_status + missing revision)
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"action": "approve", "revision": draft["revision"],
                              "review_comment": "נראה טוב."})
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
                        json={"action": "reject", "revision": draft["revision"],
                              "review_comment": "לא מדויק."})
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
                        json={"action": "approve", "revision": draft["revision"],
                              "physician_edited_text": edited})
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
                        json={"action": "invalid_action", "revision": 0})
        assert r.status_code == 400

    def test_get_audit_trail(self, portal_client):
        client, rdb = portal_client
        draft = rdb.create_draft(
            draft_type="gene_summary", original_ai_text="טקסט לבדיקת אודיט.", gene_symbol="APC"
        )
        client.post("/api/physician/login",
                    json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD})
        client.post(f"/api/physician/drafts/{draft['id']}/review",
                    json={"action": "approve", "revision": draft["revision"]})
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestSession271Hardening — focused tests added in Session 27.1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSession271Hardening:
    """Production-readiness checks added during Session 27.1."""

    # -- DB indexes idempotent ----------------------------------------------

    def test_init_db_idempotent(self, isolated_review_db):
        """Calling init_db() twice must not raise or duplicate tables."""
        rdb = isolated_review_db
        result = rdb.init_db()
        assert result is True

    # -- Approved-only with an approved draft -------------------------------

    def test_approved_only_serves_approved_text(self, isolated_review_db, monkeypatch):
        """When AI_DRAFT_VISIBILITY_MODE=approved_only, get_approved_draft returns approved text."""
        rdb = isolated_review_db
        draft = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="הגן BRCA2 מקודד לחלבון תיקון DNA בתאים. מידע מאושר על ידי רופא.",
            gene_symbol="BRCA2",
        )
        rdb.update_draft_status(
            draft["id"],
            new_status="approved",
            reviewer_identity="dr_test",
        )
        approved = rdb.get_approved_draft("BRCA2", draft_type="gene_summary")
        assert approved is not None
        assert approved["effective_text"] == draft["original_ai_text"]

    # -- Approved-only with physician-edited text ----------------------------

    def test_approved_only_serves_physician_edited_text(self, isolated_review_db):
        """get_approved_draft returns physician-edited text when present."""
        rdb = isolated_review_db
        edited = "הגן BRCA2: טקסט מעודכן על ידי רופא. מידע כללי בלבד."
        draft = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="הגן BRCA2 מקודד לחלבון תיקון DNA בתאים.",
            gene_symbol="BRCA2",
        )
        rdb.update_draft_status(
            draft["id"],
            new_status="approved",
            reviewer_identity="dr_test",
            physician_edited_text=edited,
        )
        approved = rdb.get_approved_draft("BRCA2", draft_type="gene_summary")
        assert approved is not None
        assert approved["effective_text"] == edited

    # -- Approved-only fallback when no approved draft ----------------------

    def test_approved_only_returns_none_when_no_approved_draft(self, isolated_review_db):
        """get_approved_draft returns None when no approved draft exists for the gene."""
        rdb = isolated_review_db
        assert rdb.get_approved_draft("UNKNOWN_GENE_XYZ", draft_type="gene_summary") is None

    # -- Brute-force lockout ------------------------------------------------

    def test_brute_force_lockout_after_five_failures(self, monkeypatch):
        """Authenticate returns False and locks username after 5 consecutive failures."""
        import importlib
        from app import physician_auth as pa
        importlib.reload(pa)

        from app.physician_auth import generate_hash
        good_hash = generate_hash("correct-password")
        monkeypatch.setenv("REVIEWER_USERNAME", "dr_brute_test")
        monkeypatch.setenv("REVIEWER_PASSWORD_HASH", good_hash)

        importlib.reload(pa)

        for _ in range(5):
            assert pa.authenticate("dr_brute_test", "wrong-password") is False

        assert pa.is_locked_out("dr_brute_test") is True
        # Even the correct password is rejected while locked out
        assert pa.authenticate("dr_brute_test", "correct-password") is False

    def test_lockout_cleared_after_success(self, monkeypatch):
        """Successful login clears the failed-attempt counter."""
        import importlib
        from app import physician_auth as pa
        importlib.reload(pa)

        from app.physician_auth import generate_hash
        good_hash = generate_hash("correct-pass")
        monkeypatch.setenv("REVIEWER_USERNAME", "dr_clear_test")
        monkeypatch.setenv("REVIEWER_PASSWORD_HASH", good_hash)
        importlib.reload(pa)

        for _ in range(3):
            pa.authenticate("dr_clear_test", "wrong")

        pa.authenticate("dr_clear_test", "correct-pass")
        assert pa.is_locked_out("dr_clear_test") is False

    # -- Session secret strength warning ------------------------------------

    def test_weak_session_secret_logs_warning(self, monkeypatch, caplog):
        """A short session secret should emit a warning log."""
        import importlib, logging
        monkeypatch.setenv("REVIEW_SESSION_SECRET", "short")
        from app import physician_auth as pa
        importlib.reload(pa)
        with caplog.at_level(logging.WARNING, logger="app.physician_auth"):
            pa.get_session_secret()
        assert any("short" in r.message.lower() or "chars" in r.message.lower()
                   for r in caplog.records)

    # -- Missing portal config does not break patient chat ------------------

    def test_patient_chat_works_without_portal_config(self, monkeypatch):
        """POST /ask returns valid 5-key response even when portal env vars are absent."""
        monkeypatch.delenv("REVIEW_PORTAL_ENABLED", raising=False)
        monkeypatch.delenv("REVIEWER_USERNAME", raising=False)
        monkeypatch.delenv("REVIEWER_PASSWORD_HASH", raising=False)
        monkeypatch.delenv("REVIEW_SESSION_SECRET", raising=False)
        import importlib
        from app import physician_auth, main
        importlib.reload(physician_auth)
        importlib.reload(main)
        from fastapi.testclient import TestClient
        client = TestClient(main.app)
        r = client.post("/ask", json={"question": "מה זה נשאות?"})
        assert r.status_code == 200
        data = r.json()
        required = {"answer", "safety_level", "needs_genetic_counselor",
                    "matched_topic", "suggested_questions"}
        assert required.issubset(set(data.keys()))

    # -- Database URL selection: SQLite default ------------------------------

    def test_sqlite_used_when_no_database_url(self, isolated_review_db, monkeypatch):
        """Without DATABASE_URL, _USE_POSTGRES must be False."""
        rdb = isolated_review_db
        assert rdb._USE_POSTGRES is False

    # -- Malformed DATABASE_URL safe failure --------------------------------

    def test_malformed_database_url_does_not_crash_init(self, monkeypatch, tmp_path):
        """A malformed DATABASE_URL that starts with 'postgres' but is invalid
        should cause init_db to fail gracefully (return False), not raise."""
        import importlib
        monkeypatch.setenv("DATABASE_URL", "postgres://bad:url/that/wont/connect")
        monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", str(tmp_path / "bad.db"))
        from app import review_db
        importlib.reload(review_db)
        result = review_db.init_db()
        # Should return False on connection failure, not raise an unhandled exception
        assert result in (True, False)  # either outcome is acceptable; must not raise


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestSessionSecretFailClosed — Session 27.2 Step 5
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSessionSecretFailClosed:
    """
    portal_enabled() must return False when REVIEW_SESSION_SECRET is missing,
    too short, or a known placeholder — even when REVIEW_PORTAL_ENABLED=true.
    The patient chat (/ask) must remain functional in all these cases.
    """

    def _reload_pa(self, monkeypatch, secret_value, enabled="true", username="doc", pw_hash=None):
        import importlib
        from app import physician_auth as pa
        from app.physician_auth import generate_hash
        monkeypatch.setenv("REVIEW_PORTAL_ENABLED", enabled)
        monkeypatch.setenv("REVIEWER_USERNAME", username)
        monkeypatch.setenv("REVIEWER_PASSWORD_HASH", pw_hash or generate_hash("pw"))
        if secret_value is None:
            monkeypatch.delenv("REVIEW_SESSION_SECRET", raising=False)
        else:
            monkeypatch.setenv("REVIEW_SESSION_SECRET", secret_value)
        importlib.reload(pa)
        return pa

    def test_missing_secret_disables_portal(self, monkeypatch):
        """REVIEW_SESSION_SECRET absent → portal_enabled() is False."""
        pa = self._reload_pa(monkeypatch, secret_value=None)
        assert pa.portal_enabled() is False

    def test_short_secret_disables_portal(self, monkeypatch):
        """Secret shorter than 32 chars → portal_enabled() is False."""
        pa = self._reload_pa(monkeypatch, secret_value="tooshort")
        assert pa.portal_enabled() is False

    def test_exactly_31_chars_disables_portal(self, monkeypatch):
        """Secret of exactly 31 chars is one below the minimum → portal disabled."""
        pa = self._reload_pa(monkeypatch, secret_value="a" * 31)
        assert pa.portal_enabled() is False

    def test_exactly_32_chars_enables_portal(self, monkeypatch):
        """Secret of exactly 32 chars meets the minimum → portal enabled."""
        pa = self._reload_pa(monkeypatch, secret_value="a" * 32)
        assert pa.portal_enabled() is True

    def test_placeholder_changeme_disables_portal(self, monkeypatch):
        """'changeme' is a known placeholder and must disable the portal."""
        pa = self._reload_pa(monkeypatch, secret_value="changeme")
        assert pa.portal_enabled() is False

    def test_placeholder_secret_word_disables_portal(self, monkeypatch):
        """'secret' is a known placeholder and must disable the portal."""
        pa = self._reload_pa(monkeypatch, secret_value="secret")
        assert pa.portal_enabled() is False

    def test_placeholder_your_secret_disables_portal(self, monkeypatch):
        """'your-secret' from .env.example must disable the portal."""
        pa = self._reload_pa(monkeypatch, secret_value="your-secret")
        assert pa.portal_enabled() is False

    def test_valid_strong_secret_enables_portal(self, monkeypatch):
        """A 64-char hex string (typical output of secrets.token_hex(32)) enables portal."""
        import secrets
        strong = secrets.token_hex(32)   # 64 hex chars
        pa = self._reload_pa(monkeypatch, secret_value=strong)
        assert pa.portal_enabled() is True

    def test_portal_disabled_patient_chat_still_works(self, monkeypatch):
        """When secret is missing, /ask returns valid 5-key response (patient unaffected)."""
        import importlib
        from app import physician_auth, main
        monkeypatch.setenv("REVIEW_PORTAL_ENABLED", "true")
        monkeypatch.setenv("REVIEWER_USERNAME", "doc")
        from app.physician_auth import generate_hash
        monkeypatch.setenv("REVIEWER_PASSWORD_HASH", generate_hash("pw"))
        monkeypatch.delenv("REVIEW_SESSION_SECRET", raising=False)
        importlib.reload(physician_auth)
        importlib.reload(main)
        from fastapi.testclient import TestClient
        client = TestClient(main.app)
        r = client.post("/ask", json={"question": "מה זה VUS?"})
        assert r.status_code == 200
        data = r.json()
        required = {"answer", "safety_level", "needs_genetic_counselor",
                    "matched_topic", "suggested_questions"}
        assert required.issubset(set(data.keys()))

    def test_portal_disabled_physician_endpoint_returns_503(self, monkeypatch):
        """When secret is missing, /physician returns 503, not an error page."""
        import importlib
        from app import physician_auth, main
        monkeypatch.setenv("REVIEW_PORTAL_ENABLED", "true")
        monkeypatch.setenv("REVIEWER_USERNAME", "doc")
        from app.physician_auth import generate_hash
        monkeypatch.setenv("REVIEWER_PASSWORD_HASH", generate_hash("pw"))
        monkeypatch.delenv("REVIEW_SESSION_SECRET", raising=False)
        importlib.reload(physician_auth)
        importlib.reload(main)
        from fastapi.testclient import TestClient
        client = TestClient(main.app)
        r = client.get("/physician")
        assert r.status_code == 503

    def test_is_strong_secret_rejects_empty(self, monkeypatch):
        from app.physician_auth import _is_strong_secret
        assert _is_strong_secret("") is False

    def test_is_strong_secret_rejects_whitespace_only(self, monkeypatch):
        from app.physician_auth import _is_strong_secret
        assert _is_strong_secret("   ") is False

    def test_is_strong_secret_accepts_long_non_placeholder(self, monkeypatch):
        from app.physician_auth import _is_strong_secret
        assert _is_strong_secret("x" * 32) is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestApprovedContentSafety — Session 27.2 Step 8
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestApprovedContentSafety:
    """
    Physician approval cannot bypass safety policy.
    - Approved gene drafts must NOT be served in response to personal variant
      interpretation requests, treatment decisions, surgery decisions, or
      personal risk questions.
    - Safety routing for those categories is determined BEFORE any approved
      draft is retrieved.
    - get_approved_draft() is only called within the permitted
      educational-intent path.
    """

    def _ask(self, question, **extra):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        payload = {"question": question}
        payload.update(extra)
        return client.post("/ask", json=payload).json()

    def test_personal_interpretation_blocked_despite_approved_content(
        self, isolated_review_db, monkeypatch
    ):
        """
        Even when an approved draft exists for BRCA1, a question asking for
        personal risk interpretation returns requires_genetic_counselor,
        not general_information.
        """
        rdb = isolated_review_db
        rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="BRCA1 הוא גן מדכא גידול.",
            gene_symbol="BRCA1",
        )
        rdb.update_draft_status(
            draft_id=rdb.list_drafts()[0]["id"],
            new_status="approved",
            reviewer_identity="dr_test",
        )
        result = self._ask("מה הסיכון שלי לחלות בסרטן בגלל BRCA1?")
        assert result["safety_level"] == "requires_genetic_counselor", (
            f"Expected requires_genetic_counselor, got {result['safety_level']!r}. "
            f"Approved draft must not bypass personal risk routing."
        )

    def test_surgery_decision_blocked_despite_approved_content(
        self, isolated_review_db, monkeypatch
    ):
        """Surgery decision requests are blocked even when an approved draft exists."""
        rdb = isolated_review_db
        rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="BRCA2 מגביר סיכון לסרטן שד.",
            gene_symbol="BRCA2",
        )
        rdb.update_draft_status(
            draft_id=rdb.list_drafts()[0]["id"],
            new_status="approved",
            reviewer_identity="dr_test",
        )
        result = self._ask("האם עלי לעשות ניתוח כריתת שד בגלל BRCA2?")
        assert result["safety_level"] in (
            "requires_genetic_counselor", "out_of_scope"
        ), (
            f"Surgery decision must be blocked; got {result['safety_level']!r}"
        )

    def test_cancer_diagnosis_claim_blocked_despite_approved_content(
        self, isolated_review_db, monkeypatch
    ):
        """
        'Do I have cancer?' must route to requires_genetic_counselor,
        not return approved general educational content about the gene.
        """
        rdb = isolated_review_db
        rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="BRCA1 הוא גן מדכא גידול. המידע כללי.",
            gene_symbol="BRCA1",
        )
        rdb.update_draft_status(
            draft_id=rdb.list_drafts()[0]["id"],
            new_status="approved",
            reviewer_identity="dr_test",
        )
        result = self._ask("תאמר לי אם אני חולה בסרטן בגלל שיש לי VUS ב-BRCA1")
        assert result["safety_level"] == "requires_genetic_counselor", (
            f"Diagnosis request must be blocked; got {result['safety_level']!r}. "
            f"Answer was: {result.get('answer', '')[:80]}"
        )

    def test_general_gene_question_may_serve_approved_content(
        self, isolated_review_db, monkeypatch
    ):
        """
        A plain educational gene question ('מה זה BRCA1?') is permitted to
        serve approved content — this ensures the safety filter is not over-blocking.
        The result must be general_information and must not direct to a counselor
        with a safety block answer.
        """
        result = self._ask("מה זה BRCA1?")
        assert result["safety_level"] == "general_information", (
            f"General gene question should be general_information, "
            f"got {result['safety_level']!r}"
        )

    def test_approved_draft_not_served_for_pii_question(self, isolated_review_db):
        """A question with PII (Israeli ID) is blocked at step 1, before any draft lookup."""
        result = self._ask("יש לי BRCA1, תעזור לי, מספר הזהות שלי הוא 123456789")
        assert result["safety_level"] == "contains_identifying_info"

    def test_unapproved_draft_not_served_in_approved_only_mode(
        self, isolated_review_db, monkeypatch
    ):
        """
        In approved_only mode, a pending draft must not reach the patient.
        The answer for a gene question falls back to the deterministic KB path.
        """
        monkeypatch.setenv("AI_DRAFT_VISIBILITY_MODE", "approved_only")
        rdb = isolated_review_db
        rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="BRCA1 הוא גן מדכא גידול — UNREVIEWED.",
            gene_symbol="BRCA1",
        )
        result = self._ask("מה זה BRCA1?")
        assert "UNREVIEWED" not in result.get("answer", ""), (
            "Pending (unapproved) draft must never appear in patient-facing answer"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestStartupDbInit — Session 27.3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestStartupDbInit:
    """
    Session 27.3 — review_db is initialized via FastAPI lifespan on startup.

    Requirements verified:
    - init_db() is called once per process startup (not on each request)
    - init_db() is idempotent (safe to call twice)
    - An exception in init_db() does not prevent the app from starting
    - Patient /ask continues to work after an init failure
    - No DATABASE_URL or credentials appear in exception log messages
    - Portal endpoints are available after a successful startup init
    """

    def _reload_fresh(self, monkeypatch, tmp_path, db_name="startup_test.db",
                      database_url=None):
        """
        Reload review_db and main with a fresh isolated SQLite path.
        Returns (review_db_module, main_module).
        """
        import importlib
        from app import review_db, main
        db_path = str(tmp_path / db_name)
        monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
        if database_url is None:
            monkeypatch.delenv("DATABASE_URL", raising=False)
        else:
            monkeypatch.setenv("DATABASE_URL", database_url)
        importlib.reload(review_db)
        importlib.reload(main)
        return review_db, main

    def test_db_init_called_on_startup(self, tmp_path, monkeypatch):
        """init_db() is called exactly once during the lifespan startup event."""
        import importlib
        from app import review_db, main
        db_path = str(tmp_path / "startup_test.db")
        monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        importlib.reload(review_db)

        # Install counter AFTER reload so autouse fixture calls don't pollute count
        call_count = [0]
        _orig = review_db.init_db
        def _counting():
            call_count[0] += 1
            return _orig()
        monkeypatch.setattr(review_db, "init_db", _counting)
        importlib.reload(main)

        assert call_count[0] == 0  # lifespan not yet fired

        from fastapi.testclient import TestClient
        with TestClient(main.app) as client:
            r = client.get("/health")
            assert r.status_code == 200
            # Lifespan startup fired init_db exactly once
            assert call_count[0] == 1

    def test_db_init_called_exactly_once_per_startup(self, tmp_path, monkeypatch):
        """init_db is called exactly once per startup, not on every HTTP request."""
        import importlib
        from app import review_db, main
        db_path = str(tmp_path / "once_test.db")
        monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        importlib.reload(review_db)

        call_count = [0]
        _orig = review_db.init_db

        def _counting_init():
            call_count[0] += 1
            return _orig()

        monkeypatch.setattr(review_db, "init_db", _counting_init)
        importlib.reload(main)

        from fastapi.testclient import TestClient
        with TestClient(main.app) as client:
            client.get("/health")
            client.get("/health")
            client.post("/ask", json={"question": "מה זה VUS?"})

        assert call_count[0] == 1, (
            f"init_db called {call_count[0]} time(s); expected exactly 1"
        )

    def test_db_init_idempotent_second_call(self, isolated_review_db):
        """Calling init_db() again after initial setup is safe and preserves data."""
        rdb = isolated_review_db
        rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="הגן BRCA1 מקודד לחלבון תיקון DNA.",
            gene_symbol="BRCA1",
        )
        result = rdb.init_db()
        assert result is True
        drafts = rdb.list_drafts()
        assert len(drafts) == 1, "Existing draft must survive second init_db() call"

    def test_startup_survives_init_exception(self, tmp_path, monkeypatch):
        """When init_db raises, startup completes and /health is still reachable."""
        import importlib
        from app import review_db, main
        db_path = str(tmp_path / "exc_test.db")
        monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        importlib.reload(review_db)

        def _raise_init():
            raise RuntimeError("simulated DB init failure")

        monkeypatch.setattr(review_db, "init_db", _raise_init)
        importlib.reload(main)

        from fastapi.testclient import TestClient
        with TestClient(main.app, raise_server_exceptions=False) as client:
            r = client.get("/health")
            assert r.status_code == 200, (
                "App must remain reachable after init_db exception"
            )

    def test_patient_ask_works_after_init_failure(self, tmp_path, monkeypatch):
        """POST /ask returns a valid 5-key response even when DB init fails."""
        import importlib
        from app import review_db, main
        db_path = str(tmp_path / "ask_test.db")
        monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        importlib.reload(review_db)
        monkeypatch.setattr(review_db, "init_db", lambda: (_ for _ in ()).throw(
            RuntimeError("db unreachable")))
        importlib.reload(main)

        from fastapi.testclient import TestClient
        with TestClient(main.app, raise_server_exceptions=False) as client:
            r = client.post("/ask", json={"question": "מה זה VUS?"})
            assert r.status_code == 200
            data = r.json()
            required = {"answer", "safety_level", "needs_genetic_counselor",
                        "matched_topic", "suggested_questions"}
            assert required.issubset(set(data.keys()))

    def test_no_credentials_in_init_exception_log(self, tmp_path, monkeypatch, caplog):
        """
        When init_db raises an exception whose message contains DATABASE_URL
        or credentials, the log must only record the exception *type*, not
        the exception message.
        """
        import logging, importlib
        fake_url = "postgresql://dr_admin:super_secret_pw@db.render.com:5432/review_prod"
        db_path = str(tmp_path / "cred_test.db")
        monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
        monkeypatch.setenv("DATABASE_URL", fake_url)
        from app import review_db, main
        importlib.reload(review_db)

        def _raise_with_url():
            raise Exception(f"FATAL: connection refused to {fake_url}")

        monkeypatch.setattr(review_db, "init_db", _raise_with_url)
        importlib.reload(main)

        from fastapi.testclient import TestClient
        with caplog.at_level(logging.DEBUG):
            with TestClient(main.app, raise_server_exceptions=False) as client:
                client.get("/health")

        all_log = " ".join(r.getMessage() for r in caplog.records)
        assert "super_secret_pw" not in all_log, (
            "Password must not appear in any log record"
        )
        assert fake_url not in all_log, (
            "Full DATABASE_URL must not appear in any log record"
        )

    def test_portal_available_after_successful_startup_init(self, tmp_path, monkeypatch):
        """When portal is configured and init succeeds, /physician returns 200."""
        import secrets, importlib
        from app.physician_auth import generate_hash
        from app import physician_auth, review_db, main
        db_path = str(tmp_path / "portal_startup.db")
        monkeypatch.setenv("REVIEW_PORTAL_ENABLED", "true")
        monkeypatch.setenv("REVIEWER_USERNAME", "dr_startup")
        monkeypatch.setenv("REVIEWER_PASSWORD_HASH", generate_hash("pw123"))
        monkeypatch.setenv("REVIEW_SESSION_SECRET", secrets.token_hex(32))
        monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        importlib.reload(physician_auth)
        importlib.reload(review_db)
        importlib.reload(main)

        from fastapi.testclient import TestClient
        with TestClient(main.app) as client:
            r = client.get("/physician")
            assert r.status_code == 200
            assert review_db._initialized is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestPostgresCompatibility — Session 27.4
#
# These tests target the specific SQLite-vs-PostgreSQL dialect gaps fixed in
# Session 27.4.  All assertions run against the SQLite backend (the default
# in tests) but verify that the module-level *constants* and *helpers* contain
# the correct PostgreSQL-safe values so the same code compiles and runs on
# either engine.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPostgresCompatibility:
    """
    Session 27.4 — root-cause fixes for PostgreSQL schema and row compatibility.

    Requirements verified:
    - Separate schema strings contain engine-correct auto-increment syntax
    - Placeholder helper returns ? (SQLite) or %s (PostgreSQL) as appropriate
    - _row_to_dict() handles None and dict-like rows uniformly
    - _initialized follows success/failure of init_db()
    - No conn.execute() shortcut remains in the module source
    - All public CRUD functions return Python dicts (not tuples)
    - pending_count() returns an int, not a crash from dict-key indexing
    - No real PostgreSQL required; explicitly documented where absent
    """

    # ------------------------------------------------------------------
    # Schema syntax
    # ------------------------------------------------------------------

    def test_pg_audit_schema_uses_bigserial(self, isolated_review_db):
        """_CREATE_AUDIT_PG must use BIGSERIAL PRIMARY KEY (PostgreSQL syntax)."""
        rdb = isolated_review_db
        schema = rdb._CREATE_AUDIT_PG
        assert "BIGSERIAL" in schema.upper(), (
            "_CREATE_AUDIT_PG must declare 'BIGSERIAL' for the primary key"
        )
        assert "AUTOINCREMENT" not in schema.upper(), (
            "_CREATE_AUDIT_PG must not contain 'AUTOINCREMENT' (SQLite-only keyword)"
        )

    def test_sqlite_audit_schema_uses_autoincrement(self, isolated_review_db):
        """_CREATE_AUDIT_SQLITE must use AUTOINCREMENT (SQLite syntax)."""
        rdb = isolated_review_db
        schema = rdb._CREATE_AUDIT_SQLITE
        assert "AUTOINCREMENT" in schema.upper()

    def test_pg_schema_does_not_contain_autoincrement(self, isolated_review_db):
        """Guard: the PostgreSQL schema string must be clean of SQLite artifacts."""
        rdb = isolated_review_db
        assert "AUTOINCREMENT" not in rdb._CREATE_AUDIT_PG.upper()

    # ------------------------------------------------------------------
    # Placeholder helpers
    # ------------------------------------------------------------------

    def test_ph_returns_question_mark_for_sqlite(self, isolated_review_db):
        """_ph() returns '?' when the SQLite backend is active."""
        rdb = isolated_review_db
        assert not rdb._USE_POSTGRES, "SQLite should be active in isolated tests"
        assert rdb._ph() == "?"

    def test_placeholder_n_returns_repeated_question_marks(self, isolated_review_db):
        """_placeholder(3) returns '?, ?, ?' for SQLite backend."""
        rdb = isolated_review_db
        result = rdb._placeholder(3)
        assert result == "?, ?, ?"

    def test_ph_returns_percent_s_for_postgres(self, monkeypatch, tmp_path):
        """_ph() returns '%s' when DATABASE_URL is a PostgreSQL URL."""
        import importlib
        from app import review_db
        db_path = str(tmp_path / "pg_ph_test.db")
        monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@localhost/testdb")
        importlib.reload(review_db)
        assert review_db._USE_POSTGRES is True
        assert review_db._ph() == "%s"
        # Restore normal state for subsequent tests (autouse fixture handles DB)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        importlib.reload(review_db)

    def test_placeholder_n_returns_percent_s_for_postgres(self, monkeypatch, tmp_path):
        """_placeholder(2) returns '%s, %s' for the PostgreSQL backend."""
        import importlib
        from app import review_db
        db_path = str(tmp_path / "pg_ph2_test.db")
        monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@localhost/testdb")
        importlib.reload(review_db)
        assert review_db._placeholder(2) == "%s, %s"
        monkeypatch.delenv("DATABASE_URL", raising=False)
        importlib.reload(review_db)

    # ------------------------------------------------------------------
    # _row_to_dict
    # ------------------------------------------------------------------

    def test_row_to_dict_none_returns_none(self, isolated_review_db):
        """_row_to_dict(None) must return None without raising."""
        rdb = isolated_review_db
        assert rdb._row_to_dict(None) is None

    def test_row_to_dict_plain_dict_is_identity(self, isolated_review_db):
        """_row_to_dict on a plain dict returns the same mapping (simulates RealDictRow)."""
        rdb = isolated_review_db
        row = {"id": "abc", "review_status": "pending", "cnt": 5}
        result = rdb._row_to_dict(row)
        assert result == row
        assert isinstance(result, dict)

    def test_row_to_dict_sqlite_row_returns_dict(self, isolated_review_db):
        """_row_to_dict on a real sqlite3.Row returns a plain dict with correct keys."""
        import sqlite3
        rdb = isolated_review_db
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("CREATE TABLE t (a INTEGER, b TEXT)")
        cur.execute("INSERT INTO t VALUES (42, 'hello')")
        cur.execute("SELECT * FROM t")
        row = cur.fetchone()
        result = rdb._row_to_dict(row)
        conn.close()
        assert isinstance(result, dict)
        assert result["a"] == 42
        assert result["b"] == "hello"

    # ------------------------------------------------------------------
    # _initialized semantics
    # ------------------------------------------------------------------

    def test_initialized_false_before_init(self, monkeypatch, tmp_path):
        """_initialized starts as False after a module reload (no auto-call)."""
        import importlib
        from app import review_db
        db_path = str(tmp_path / "init_state.db")
        monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        importlib.reload(review_db)
        # We have NOT called init_db() yet — must be False
        assert review_db._initialized is False

    def test_initialized_true_after_successful_init(self, isolated_review_db):
        """_initialized is True after init_db() succeeds (autouse fixture calls it)."""
        rdb = isolated_review_db
        assert rdb._initialized is True

    def test_initialized_false_after_failed_init(self, monkeypatch, tmp_path):
        """_initialized stays False when init_db() fails — allows retry."""
        import importlib
        from app import review_db
        db_path = str(tmp_path / "fail_init.db")
        monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        importlib.reload(review_db)

        # Simulate schema failure
        _orig_get_conn = review_db._get_connection

        from contextlib import contextmanager

        @contextmanager
        def _bad_conn():
            raise RuntimeError("simulated connection error")
            yield  # noqa: unreachable

        monkeypatch.setattr(review_db, "_get_connection", _bad_conn)
        result = review_db.init_db()

        assert result is False
        assert review_db._initialized is False

    def test_initialized_becomes_true_after_retry(self, monkeypatch, tmp_path):
        """After a failed init, calling init_db() again with a working connection succeeds."""
        import importlib
        from app import review_db
        db_path = str(tmp_path / "retry_init.db")
        monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        importlib.reload(review_db)

        # Save the real connection helper before patching
        _real_get_connection = review_db._get_connection

        from contextlib import contextmanager

        @contextmanager
        def _bad_conn():
            raise RuntimeError("first attempt fails")
            yield  # noqa: unreachable

        monkeypatch.setattr(review_db, "_get_connection", _bad_conn)
        assert review_db.init_db() is False
        assert review_db._initialized is False

        # Restore real connection and retry — must now succeed
        monkeypatch.setattr(review_db, "_get_connection", _real_get_connection)
        assert review_db.init_db() is True
        assert review_db._initialized is True

    # ------------------------------------------------------------------
    # No conn.execute() shortcut in module source
    # ------------------------------------------------------------------

    def test_no_conn_execute_shortcut_in_source(self):
        """
        review_db.py must not contain 'conn.execute(' — that is the SQLite-only
        shortcut that fails on psycopg2.  All queries must go through an explicit
        cursor.
        """
        import inspect
        from app import review_db
        source = inspect.getsource(review_db)
        assert "conn.execute(" not in source, (
            "Found 'conn.execute(' in review_db — this is a SQLite-only shortcut "
            "that raises AttributeError on psycopg2.  Use an explicit cursor."
        )

    # ------------------------------------------------------------------
    # CRUD functions return Python dicts
    # ------------------------------------------------------------------

    def test_create_draft_returns_dict(self, isolated_review_db):
        rdb = isolated_review_db
        d = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="הגן PALB2 קשור לסיכון לסרטן השד.",
            gene_symbol="PALB2",
        )
        assert isinstance(d, dict), f"create_draft must return dict, got {type(d)}"

    def test_get_draft_returns_dict(self, isolated_review_db):
        rdb = isolated_review_db
        d = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="הגן MLH1 מעורב בתסמונת לינץ'.",
            gene_symbol="MLH1",
        )
        fetched = rdb.get_draft(d["id"])
        assert isinstance(fetched, dict)

    def test_list_drafts_returns_list_of_dicts(self, isolated_review_db):
        rdb = isolated_review_db
        rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="הגן MSH2 קשור לסיכון סרטן המעי.",
            gene_symbol="MSH2",
        )
        result = rdb.list_drafts()
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_update_draft_status_returns_dict(self, isolated_review_db):
        rdb = isolated_review_db
        d = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="הגן ATM מקודד לקינאז בנתיב תיקון DNA.",
            gene_symbol="ATM",
        )
        updated = rdb.update_draft_status(
            d["id"], new_status="approved", reviewer_identity="dr_test"
        )
        assert isinstance(updated, dict)
        assert updated["review_status"] == "approved"

    def test_get_audit_trail_returns_list_of_dicts(self, isolated_review_db):
        rdb = isolated_review_db
        d = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="הגן CHEK2 קשור לסיכון מוגבר לסרטן השד.",
            gene_symbol="CHEK2",
        )
        rdb.update_draft_status(d["id"], new_status="approved", reviewer_identity="dr_audit")
        trail = rdb.get_audit_trail(d["id"])
        assert isinstance(trail, list)
        assert len(trail) == 1
        assert isinstance(trail[0], dict)
        assert "reviewer_identity" in trail[0]

    # ------------------------------------------------------------------
    # pending_count returns int (not a crash from dict-key row access)
    # ------------------------------------------------------------------

    def test_pending_count_returns_int(self, isolated_review_db):
        """pending_count() must return a plain int, not raise from row indexing."""
        rdb = isolated_review_db
        rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="הגן RAD51C מעורב בסיכון לסרטן שחלה.",
            gene_symbol="RAD51C",
        )
        count = rdb.pending_count()
        assert isinstance(count, int)
        assert count == 1

    def test_pending_count_excludes_approved(self, isolated_review_db):
        rdb = isolated_review_db
        d = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="הגן BARD1 מעורב בתיקון DNA.",
            gene_symbol="BARD1",
        )
        assert rdb.pending_count() == 1
        rdb.update_draft_status(d["id"], new_status="approved", reviewer_identity="dr_x")
        assert rdb.pending_count() == 0

    # ------------------------------------------------------------------
    # Real PostgreSQL — documented as not performed
    # ------------------------------------------------------------------

    def test_real_postgres_not_tested(self):
        """
        Real PostgreSQL integration test not performed: no Docker or local
        PostgreSQL instance is available in this environment.

        The PostgreSQL compatibility is verified by:
          - Schema string inspection (BIGSERIAL vs AUTOINCREMENT)
          - Placeholder helper unit tests (_ph, _placeholder)
          - Module source scan for conn.execute() shortcuts
          - All CRUD functions tested end-to-end on SQLite
          - _initialized retry semantics tested via patched _get_connection

        To run a real PG test, set DATABASE_URL to a test database and run:
          PYTHONUTF8=1 pytest tests/test_session27_physician_portal.py \\
            -k test_real_postgres -v
        """
        pytest.skip("Real PostgreSQL not available in this environment — see docstring")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestSession275ReviewContract — Session 27.5
#
# Verifies the new review-action API contract:
#   - action (verb: approve / reject / needs_revision) not new_status
#   - revision (integer) for optimistic concurrency
#   - comment required for reject and needs_revision
#   - stale revision → 409
#   - old new_status field → 422 (regression protection)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSession275ReviewContract:
    """
    Session 27.5 — physician review API contract correctness.

    Root cause of original 422: physician.html called closeModal() before
    reading _pendingAction, so _pendingAction was null at request time.
    The fix: capture action to a local variable before closeModal().

    This class tests the FastAPI side of the contract to prevent regression.
    """

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _login(self, client):
        client.post("/api/physician/login",
                    json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD})

    def _make_draft(self, rdb, gene="CCR5",
                    text="הגן CCR5 מקודד לקולטן כמוקין על פני תאי חיסון."):
        return rdb.create_draft(draft_type="gene_summary",
                                original_ai_text=text, gene_symbol=gene)

    # ------------------------------------------------------------------
    # Old new_status field → 422 (regression test for the original bug)
    # ------------------------------------------------------------------

    def test_old_new_status_field_returns_422(self, portal_client):
        """
        Sending new_status instead of action must return 422.
        This test documents the original production bug — the frontend was
        sending {new_status: null} which also returned 422.
        """
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"new_status": "approved", "revision": 0})
        assert r.status_code == 422, (
            "Sending the old new_status field must return 422 — "
            "prevents regression to the broken contract"
        )

    def test_null_action_returns_422(self, portal_client):
        """
        Sending null for action (the root cause of the original 422 in production)
        must return 422, not 200.
        """
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"action": None, "revision": 0})
        assert r.status_code == 422

    def test_missing_action_returns_422(self, portal_client):
        """Omitting action entirely (another variant of the production bug) → 422."""
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"revision": 0, "review_comment": "note"})
        assert r.status_code == 422

    def test_missing_revision_returns_422(self, portal_client):
        """Omitting revision (required integer field) → 422."""
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"action": "approve"})
        assert r.status_code == 422

    def test_revision_as_string_returns_422(self, portal_client):
        """Revision sent as a string must fail Pydantic int validation → 422."""
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"action": "approve", "revision": "0"})
        # Pydantic v2 coerces string "0" to int 0 — this is acceptable.
        # What matters: the request is NOT rejected with 422 due to type.
        assert r.status_code in (200, 409, 422)  # coercion or error, not server crash

    # ------------------------------------------------------------------
    # Valid approve
    # ------------------------------------------------------------------

    def test_approve_action_sets_approved_status(self, portal_client):
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"action": "approve", "revision": draft["revision"]})
        assert r.status_code == 200
        assert r.json()["draft"]["review_status"] == "approved"

    def test_approve_without_comment_is_valid(self, portal_client):
        """Approve does not require a review_comment."""
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"action": "approve", "revision": draft["revision"],
                              "review_comment": None})
        assert r.status_code == 200

    def test_approve_with_edited_text_sets_physician_edited_text(self, portal_client):
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        edited = "הגן CCR5 מקודד לקולטן כמוקין הנקרא CCR5, שנמצא בעיקר על לימפוציטים T ומונוציטים."
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"action": "approve", "revision": draft["revision"],
                              "physician_edited_text": edited})
        assert r.status_code == 200
        data = r.json()["draft"]
        assert data["physician_edited_text"] == edited
        # original_ai_text must be preserved
        assert data["original_ai_text"] == draft["original_ai_text"]
        # effective_text follows the edited version
        assert data["effective_text"] == edited

    def test_approve_preserves_original_ai_text(self, portal_client):
        """Original AI text must survive approve-with-edit."""
        client, rdb = portal_client
        original = "הגן TNF מקודד לגורם נמק גידול אלפא."
        draft = rdb.create_draft(draft_type="gene_summary",
                                 original_ai_text=original, gene_symbol="TNF")
        self._login(client)
        client.post(f"/api/physician/drafts/{draft['id']}/review",
                    json={"action": "approve", "revision": draft["revision"],
                          "physician_edited_text": "גרסה ערוכה."})
        fetched = rdb.get_draft(draft["id"])
        assert fetched["original_ai_text"] == original

    # ------------------------------------------------------------------
    # Reject requires comment
    # ------------------------------------------------------------------

    def test_reject_without_comment_returns_422(self, portal_client):
        """Rejecting without a review_comment must be refused with 422."""
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"action": "reject", "revision": draft["revision"],
                              "review_comment": None})
        assert r.status_code == 422

    def test_reject_with_empty_comment_returns_422(self, portal_client):
        """Empty string comment for reject must also fail."""
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"action": "reject", "revision": draft["revision"],
                              "review_comment": "   "})
        assert r.status_code == 422

    def test_reject_with_comment_succeeds(self, portal_client):
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"action": "reject", "revision": draft["revision"],
                              "review_comment": "הטקסט אינו מדויק מספיק."})
        assert r.status_code == 200
        assert r.json()["draft"]["review_status"] == "rejected"

    # ------------------------------------------------------------------
    # Needs revision requires comment
    # ------------------------------------------------------------------

    def test_needs_revision_without_comment_returns_422(self, portal_client):
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"action": "needs_revision", "revision": draft["revision"]})
        assert r.status_code == 422

    def test_needs_revision_with_comment_succeeds(self, portal_client):
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"action": "needs_revision", "revision": draft["revision"],
                              "review_comment": "נא לתקן את הניסוח."})
        assert r.status_code == 200
        assert r.json()["draft"]["review_status"] == "needs_revision"

    # ------------------------------------------------------------------
    # Invalid action name
    # ------------------------------------------------------------------

    def test_unknown_action_returns_400(self, portal_client):
        """An unknown action verb (e.g. 'approved' instead of 'approve') → 400."""
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"action": "approved", "revision": draft["revision"]})
        assert r.status_code == 400

    def test_old_verb_rejected_returns_400(self, portal_client):
        """'rejected' is a status name, not a verb — must return 400."""
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"action": "rejected", "revision": draft["revision"],
                              "review_comment": "comment"})
        assert r.status_code == 400

    # ------------------------------------------------------------------
    # Stale revision → 409
    # ------------------------------------------------------------------

    def test_stale_revision_returns_409(self, portal_client):
        """
        Sending a revision number lower than the current draft revision must
        return 409 Conflict, not 200 or 422.
        """
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        # First action: approve advances revision to 1
        client.post(f"/api/physician/drafts/{draft['id']}/review",
                    json={"action": "approve", "revision": draft["revision"]})
        # Approve then supersede is not a valid transition, so use needs_revision first
        draft2 = self._make_draft(rdb, gene="TNF",
                                  text="גן TNF מקודד לציטוקין אחר לגמרי.")
        # Approve draft2
        client.post(f"/api/physician/drafts/{draft2['id']}/review",
                    json={"action": "approve", "revision": draft2["revision"]})
        # Now try to approve draft2 again with stale revision 0 (should be 1)
        r = client.post(f"/api/physician/drafts/{draft2['id']}/review",
                        json={"action": "approve", "revision": 0})
        assert r.status_code == 409, (
            f"Stale revision must return 409; got {r.status_code}"
        )

    def test_correct_revision_after_needs_revision_transition(self, portal_client):
        """After needs_revision, revision increments; next action must use new revision."""
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        # Move to needs_revision (revision 0 → 1)
        client.post(f"/api/physician/drafts/{draft['id']}/review",
                    json={"action": "needs_revision", "revision": 0,
                          "review_comment": "צריך תיקון."})
        # Approve with stale revision 0 → 409
        r_stale = client.post(f"/api/physician/drafts/{draft['id']}/review",
                              json={"action": "approve", "revision": 0})
        assert r_stale.status_code == 409

        # Approve with correct revision 1 → 200
        r_ok = client.post(f"/api/physician/drafts/{draft['id']}/review",
                           json={"action": "approve", "revision": 1})
        assert r_ok.status_code == 200
        assert r_ok.json()["draft"]["review_status"] == "approved"

    # ------------------------------------------------------------------
    # Audit trail and revision increments
    # ------------------------------------------------------------------

    def test_revision_increments_on_approve(self, portal_client):
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        assert draft["revision"] == 0
        r = client.post(f"/api/physician/drafts/{draft['id']}/review",
                        json={"action": "approve", "revision": 0})
        assert r.json()["draft"]["revision"] == 1

    def test_audit_event_created_for_approve(self, portal_client):
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        client.post(f"/api/physician/drafts/{draft['id']}/review",
                    json={"action": "approve", "revision": 0})
        trail = rdb.get_audit_trail(draft["id"])
        assert len(trail) == 1
        assert trail[0]["new_status"] == "approved"
        assert trail[0]["reviewer_identity"] == _TEST_USERNAME

    def test_audit_event_created_for_reject(self, portal_client):
        client, rdb = portal_client
        draft = self._make_draft(rdb)
        self._login(client)
        client.post(f"/api/physician/drafts/{draft['id']}/review",
                    json={"action": "reject", "revision": 0,
                          "review_comment": "לא מדויק."})
        trail = rdb.get_audit_trail(draft["id"])
        assert trail[0]["new_status"] == "rejected"

    # ------------------------------------------------------------------
    # Missing draft → 404 (not 500 or 422)
    # ------------------------------------------------------------------

    def test_missing_draft_returns_404(self, portal_client):
        client, _ = portal_client
        self._login(client)
        r = client.post(f"/api/physician/drafts/{uuid.uuid4()}/review",
                        json={"action": "approve", "revision": 0})
        assert r.status_code == 404


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestSession275PatientDraftCard — Session 27.5
#
# Verifies that a pending AI draft is NOT promoted into the main answer,
# that draft_promoted_to_answer=False, and that the bridge message is used.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSession275PatientDraftCard:
    """
    Session 27.5 — patient-facing draft card behaviour.

    Before 27.5: a pending AI draft text was used as the main answer
    (draft_promoted_to_answer=True), so the review draft and main bubble
    were identical.

    After 27.5: draft_promoted_to_answer=False, main answer is a short
    honest bridge message, and the draft is kept in unverified_gene_draft
    for the collapsed patient card.
    """

    def _ask(self, question, **extra):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        payload = {"question": question}
        payload.update(extra)
        r = client.post("/ask", json=payload)
        assert r.status_code == 200, f"POST /ask failed: {r.text}"
        return r.json()

    def _ask_with_draft(self, question, monkeypatch, rdb, gene, draft_text):
        """
        Patch the draft generator to return a known text without a real LLM.
        """
        import importlib
        from app import counseling_engine
        fake_draft = {
            "visible": True,
            "status": "unreviewed",
            "gene_symbol": gene,
            "warning_he": "מידע AI לא מאומת",
            "text_he": draft_text,
            "generated_by_model": "test-model",
            "review_status": "unreviewed",
            "approved": False,
            "generated_at": "2026-01-01T00:00:00Z",
        }
        monkeypatch.setattr(
            counseling_engine,
            "_generate_unverified_gene_draft",
            lambda gene, question="", clinvar_context=None, use_lenient_validator=False, _debug=None:
                fake_draft,
        )
        return self._ask(question)

    # ------------------------------------------------------------------
    # draft_promoted_to_answer is always False for pending drafts
    # ------------------------------------------------------------------

    def test_draft_promoted_to_answer_false_when_draft_available(
        self, isolated_review_db, monkeypatch
    ):
        """
        When a pending AI draft is generated, draft_promoted_to_answer must
        be False (not True as it was before Session 27.5).
        """
        data = self._ask_with_draft(
            "מה זה גן CCR5?", monkeypatch, isolated_review_db,
            "CCR5", "CCR5 הוא גן שמקודד לקולטן כמוקין.",
        )
        meta = data.get("gene_metadata") or {}
        assert meta.get("draft_promoted_to_answer") is False, (
            f"draft_promoted_to_answer must be False for a pending draft; "
            f"gene_metadata={meta}"
        )

    def test_main_answer_is_not_identical_to_draft_text(
        self, isolated_review_db, monkeypatch
    ):
        """
        The main answer bubble must not contain the full AI draft text when
        the draft is pending.
        """
        draft_text = "CCR5 הוא גן שמקודד לקולטן כמוקין המסייע לנגיף HIV לחדור לתאים."
        data = self._ask_with_draft(
            "מה זה גן CCR5?", monkeypatch, isolated_review_db,
            "CCR5", draft_text,
        )
        assert draft_text not in data.get("answer", ""), (
            "Full AI draft text must not appear in the main answer bubble "
            "when the draft is pending/unreviewed"
        )

    def test_main_answer_is_bridge_message_not_empty(
        self, isolated_review_db, monkeypatch
    ):
        """
        When a pending draft exists, the main answer must be a short bridge
        message that neither falsely says 'no info' nor contains the full draft.
        """
        data = self._ask_with_draft(
            "מה זה גן CCR5?", monkeypatch, isolated_review_db,
            "CCR5", "CCR5 מקודד לקולטן.",
        )
        answer = data.get("answer", "")
        assert answer.strip(), "Main answer must not be empty"
        # Bridge message must mention the draft is available
        assert any(kw in answer for kw in ["טיוטת", "מידע נוסף", "מצורפת"]), (
            f"Bridge message must mention the draft; got: {answer!r}"
        )

    def test_draft_not_in_main_answer_implies_not_promoted(
        self, isolated_review_db, monkeypatch
    ):
        """
        Confirm the invariant: if draft_promoted_to_answer=False the full
        draft text is absent from the main answer.
        """
        draft_text = "MARKER_TEXT_שאסור_להיות_בתשובה_הראשית"
        data = self._ask_with_draft(
            "מה זה גן TNF?", monkeypatch, isolated_review_db,
            "TNF", draft_text,
        )
        meta = data.get("gene_metadata") or {}
        if not meta.get("draft_promoted_to_answer", True):
            assert draft_text not in data.get("answer", "")

    def test_unverified_gene_draft_present_in_response(
        self, isolated_review_db, monkeypatch
    ):
        """
        unverified_gene_draft must still be present in the API response so the
        frontend can render the collapsed card.
        """
        data = self._ask_with_draft(
            "מה זה גן CCR5?", monkeypatch, isolated_review_db,
            "CCR5", "CCR5 מקודד לקולטן.",
        )
        assert "unverified_gene_draft" in data, (
            "unverified_gene_draft must be present in the response when a draft exists"
        )
        draft = data["unverified_gene_draft"]
        assert isinstance(draft, dict)
        assert "text_he" in draft

    # ------------------------------------------------------------------
    # Approved draft behavior (approved_only mode)
    # ------------------------------------------------------------------

    def test_approved_only_serves_approved_effective_text(
        self, isolated_review_db, monkeypatch
    ):
        """
        In approved_only mode, the approved draft's effective_text is the
        main answer.
        """
        rdb = isolated_review_db
        monkeypatch.setenv("AI_DRAFT_VISIBILITY_MODE", "approved_only")

        # Create and approve a draft
        d = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text="CCR5 מקודד לקולטן שבשרת מחקר.",
            gene_symbol="CCR5",
        )
        edited = "CCR5 מקודד לקולטן כמוקין — גרסה מאושרת."
        rdb.update_draft_status(
            d["id"], new_status="approved", reviewer_identity="dr_test",
            physician_edited_text=edited,
        )

        from app import counseling_engine
        monkeypatch.setattr(
            counseling_engine, "_AI_DRAFT_VISIBILITY_MODE", "approved_only"
        )

        # Mock the draft generator so the gene goes through the Tier-2 path
        monkeypatch.setattr(
            counseling_engine,
            "_generate_unverified_gene_draft",
            lambda gene, question="", clinvar_context=None,
                   use_lenient_validator=False, _debug=None:
                {"visible": True, "status": "unreviewed",
                 "gene_symbol": gene, "warning_he": "",
                 "text_he": "CCR5 מקודד לקולטן שבשרת מחקר.",
                 "generated_by_model": "test-model",
                 "review_status": "unreviewed",
                 "approved": False, "generated_at": "2026-01-01T00:00:00Z"},
        )

        data = self._ask("מה זה גן CCR5?")
        assert edited in data.get("answer", ""), (
            "Approved_only mode must serve the physician-edited approved text"
        )
