# -*- coding: utf-8 -*-
"""
Session 15 hotfix — gene-question routing regression tests.

Covers four bugs fixed in this session:
  1. "מה זה VUS בHBB" (no space, Hebrew prefix attached) was not detecting HBB.
  2. "מה המוטציה של הגן HBB" was not routing to gene-level answer.
  3. "מה זה הגן HBB" returned generic KB answer instead of HBB bio context.
  4. Gene answers for APC / BRCA1 mentioned VUS even without a VUS question.
"""

import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_local_llm(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# 1. VUS + HBB — no space between ב and HBB
# ---------------------------------------------------------------------------

class TestVusHbbNoSpace:
    def test_vus_hbb_no_space_returns_200(self):
        resp = client.post("/ask", json={"question": "מה זה VUS בHBB?"})
        assert resp.status_code == 200

    def test_vus_hbb_no_space_includes_hbb_context(self):
        resp = client.post("/ask", json={"question": "מה זה VUS בHBB?"})
        answer = resp.json()["answer"]
        assert any(term in answer for term in ("HBB", "המוגלובין", "beta-globin", "תלסמיה")), (
            f"Expected HBB biological context in answer, got: {answer[:300]}"
        )

    def test_vus_hbb_no_space_safety_level(self):
        resp = client.post("/ask", json={"question": "מה זה VUS בHBB?"})
        assert resp.json()["safety_level"] == "general_information"


# ---------------------------------------------------------------------------
# 2. VUS + HBB — with hyphen (ב-HBB)
# ---------------------------------------------------------------------------

class TestVusHbbHyphen:
    def test_vus_hbb_hyphen_returns_200(self):
        resp = client.post("/ask", json={"question": "מה זה VUS ב-HBB?"})
        assert resp.status_code == 200

    def test_vus_hbb_hyphen_includes_hbb_context(self):
        resp = client.post("/ask", json={"question": "מה זה VUS ב-HBB?"})
        answer = resp.json()["answer"]
        assert any(term in answer for term in ("HBB", "המוגלובין", "beta-globin", "תלסמיה")), (
            f"Expected HBB biological context in answer, got: {answer[:300]}"
        )

    def test_vus_hbb_hyphen_safety_level(self):
        resp = client.post("/ask", json={"question": "מה זה VUS ב-HBB?"})
        assert resp.json()["safety_level"] == "general_information"


# ---------------------------------------------------------------------------
# 3. "מה המוטציה של הגן HBB" — routes to gene-level, not generic KB
# ---------------------------------------------------------------------------

class TestMutationQuestionHbb:
    def test_mutation_question_hbb_returns_200(self):
        resp = client.post("/ask", json={"question": "מה המוטציה של הגן HBB?"})
        assert resp.status_code == 200

    def test_mutation_question_hbb_shows_gene_context(self):
        resp = client.post("/ask", json={"question": "מה המוטציה של הגן HBB?"})
        answer = resp.json()["answer"]
        assert any(term in answer for term in ("HBB", "המוגלובין", "beta-globin", "תלסמיה")), (
            f"Expected HBB biological context in mutation question answer, got: {answer[:300]}"
        )

    def test_mutation_question_does_not_claim_single_mutation(self):
        resp = client.post("/ask", json={"question": "מה המוטציה של הגן HBB?"})
        answer = resp.json()["answer"]
        forbidden = ["המוטציה היחידה של הגן", "יש מוטציה אחת", "יש וריאנט יחידי"]
        for phrase in forbidden:
            assert phrase not in answer, (
                f"Answer incorrectly claims a single mutation for HBB: '{phrase}'"
            )

    def test_mutation_question_hbb_safety_level(self):
        resp = client.post("/ask", json={"question": "מה המוטציה של הגן HBB?"})
        assert resp.json()["safety_level"] == "general_information"


# ---------------------------------------------------------------------------
# 4. "מה זה הגן HBB" — returns approved HBB summary, not generic gene KB entry
# ---------------------------------------------------------------------------

class TestPlainGeneQuestionHbb:
    def test_plain_hbb_question_returns_200(self):
        resp = client.post("/ask", json={"question": "מה זה הגן HBB?"})
        assert resp.status_code == 200

    def test_plain_hbb_question_shows_hbb_content(self):
        resp = client.post("/ask", json={"question": "מה זה הגן HBB?"})
        answer = resp.json()["answer"]
        assert any(term in answer for term in ("HBB", "המוגלובין", "beta-globin", "תלסמיה")), (
            f"Expected HBB-specific content, got: {answer[:300]}"
        )

    def test_plain_hbb_question_not_generic_gene_kb(self):
        resp = client.post("/ask", json={"question": "מה זה הגן HBB?"})
        assert resp.json().get("matched_topic") != "gene", (
            "Routed to generic 'gene' KB topic instead of gene-specific answer"
        )


# ---------------------------------------------------------------------------
# 5. "מה זה הגן APC" — answer must NOT mention VUS
# ---------------------------------------------------------------------------

class TestApcNoVus:
    def test_apc_question_returns_200(self):
        resp = client.post("/ask", json={"question": "מה זה הגן APC?"})
        assert resp.status_code == 200

    def test_apc_answer_does_not_mention_vus(self):
        resp = client.post("/ask", json={"question": "מה זה הגן APC?"})
        answer = resp.json()["answer"]
        assert "VUS" not in answer, (
            f"APC gene answer should not mention VUS when question did not ask about VUS. "
            f"Answer: {answer[:400]}"
        )

    def test_apc_answer_contains_apc_content(self):
        resp = client.post("/ask", json={"question": "מה זה הגן APC?"})
        assert "APC" in resp.json()["answer"]


# ---------------------------------------------------------------------------
# 6. "מה זה BRCA1" — answer must NOT mention VUS
# ---------------------------------------------------------------------------

class TestBrca1NoVus:
    def test_brca1_question_returns_200(self):
        resp = client.post("/ask", json={"question": "מה זה BRCA1?"})
        assert resp.status_code == 200

    def test_brca1_answer_does_not_mention_vus(self):
        resp = client.post("/ask", json={"question": "מה זה BRCA1?"})
        answer = resp.json()["answer"]
        assert "VUS" not in answer, (
            f"BRCA1 gene answer should not mention VUS when question did not ask about VUS. "
            f"Answer: {answer[:400]}"
        )

    def test_brca1_answer_contains_brca1_content(self):
        resp = client.post("/ask", json={"question": "מה זה BRCA1?"})
        assert "BRCA1" in resp.json()["answer"]
