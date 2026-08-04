# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for the genetic-bot-clinvar test suite.
"""
import pytest


@pytest.fixture(autouse=True, scope="session")
def init_review_db():
    """Initialize the SQLite review-drafts schema before any test touches the DB."""
    from app import review_db
    review_db.init_db()
