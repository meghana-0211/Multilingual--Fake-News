"""
backend/tests/conftest.py

Shared pytest fixtures.
ML model and blockchain are mocked — no GPU or Ganache needed.
"""

import sys
import os
import pytest

# Make backend the root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock
import numpy as np


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------

def _make_detector():
    det = MagicMock()
    det.predict.return_value = {
        "prediction":    "fake",
        "label":         "Fake",
        "confidence":    0.94,
        "probabilities": {"real": 0.06, "fake": 0.94},
    }
    det.predict_batch_proba.return_value = np.array([[0.06, 0.94]])
    return det


def _make_explainer():
    exp = MagicMock()
    exp.explain_comprehensive.return_value = {
        "prediction": {"label": "Fake", "confidence": 0.94, "prediction": 1},
        "lime_explanation": {
            "feature_importance": [("झूठी", -0.8), ("वायरस", 0.4)],
            "lime_score": 0.75,
        },
        "attention_explanation": {
            "top_tokens": [("वायरस", 0.6), ("पानी", 0.3)],
            "all_tokens": [("वायरस", 0.6)],
        },
        "combined_top_words": [("झूठी", 0.9), ("वायरस", 0.7)],
        "text": "test",
    }
    return exp


# ---------------------------------------------------------------------------
# App fixture (session-scoped — one DB per test session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def app():
    from backend.app import create_app

    application = create_app("development")
    application.config.update({
        "TESTING":                   True,
        "SQLALCHEMY_DATABASE_URI":   "sqlite:///:memory:",
        "DETECTOR":                  _make_detector(),
        "EXPLAINER":                 _make_explainer(),
        "PREPROCESSOR":              None,
        "BLOCKCHAIN":                None,
        "BLOCKCHAIN_ENABLED":        False,
        "JWT_SECRET_KEY":            "test-jwt-secret",
        "SECRET_KEY":                "test-secret",
    })

    from backend.database.models import db
    with application.app_context():
        db.create_all()

    yield application


@pytest.fixture
def client(app):
    return app.test_client()


# ---------------------------------------------------------------------------
# Auth header fixtures
# ---------------------------------------------------------------------------

def _register_and_login(client, email, username, password):
    client.post("/api/auth/register", json={
        "email": email, "username": username, "password": password,
    })
    r = client.post("/api/auth/login", json={
        "email": email, "password": password,
    })
    token = r.get_json().get("accessToken", "")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def user_headers(client):
    return _register_and_login(
        client, "user@test.com", "testuser", "password123"
    )


@pytest.fixture
def admin_headers(client, app):
    """Creates admin user directly in DB then logs in."""
    with app.app_context():
        from backend.database.crud   import get_user_by_email, create_user
        from backend.database.models import User, UserRole, db
        if not get_user_by_email("admin@test.com"):
            u = create_user("admin@test.com", "testadmin", "password123", role="admin")
        else:
            u = get_user_by_email("admin@test.com")
    return _register_and_login(client, "admin@test.com", "testadmin2_", "password123")


@pytest.fixture
def publisher_headers(client, app):
    with app.app_context():
        from backend.database.crud   import get_user_by_email, create_user
        from backend.database.models import User, UserRole, db
        if not get_user_by_email("pub@test.com"):
            u = create_user("pub@test.com", "testpub", "password123", role="publisher")
        else:
            u = get_user_by_email("pub@test.com")
    r = client.post("/api/auth/login", json={
        "email": "pub@test.com", "password": "password123",
    })
    token = r.get_json().get("accessToken", "")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def checker_headers(client, app):
    with app.app_context():
        from backend.database.crud   import get_user_by_email, create_user
        if not get_user_by_email("checker@test.com"):
            create_user("checker@test.com", "testchecker", "password123",
                        role="fact_checker")
    r = client.post("/api/auth/login", json={
        "email": "checker@test.com", "password": "password123",
    })
    token = r.get_json().get("accessToken", "")
    return {"Authorization": f"Bearer {token}"}