"""
backend/tests/test_api.py

Endpoint tests.  All ML + blockchain calls are mocked via conftest.py.

Run:
    cd backend
    pytest tests/test_api.py -v
"""

HINDI = (
    "कोरोना वायरस से बचने के लिए गर्म पानी पीना काफी है। "
    "विशेषज्ञों का दावा है कि यह वायरस गर्म पानी से मर जाता है।"
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_ok(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        d = r.get_json()
        assert d["status"] == "ok"
        assert "model_loaded" in d
        assert "blockchain"   in d
        assert "explainer"    in d


# ---------------------------------------------------------------------------
# Auth — register
# ---------------------------------------------------------------------------

class TestRegister:
    def test_success(self, client):
        r = client.post("/api/auth/register", json={
            "email": "reg@example.com", "username": "reguser", "password": "password123",
        })
        assert r.status_code == 201
        d = r.get_json()
        assert "accessToken" in d
        assert d["user"]["email"] == "reg@example.com"

    def test_duplicate_email(self, client):
        payload = {"email": "dup2@ex.com", "username": "dup_a", "password": "password123"}
        client.post("/api/auth/register", json=payload)
        payload["username"] = "dup_b"
        r = client.post("/api/auth/register", json=payload)
        assert r.status_code == 409

    def test_short_password(self, client):
        r = client.post("/api/auth/register", json={
            "email": "x@x.com", "username": "xu", "password": "short",
        })
        assert r.status_code == 400

    def test_missing_fields(self, client):
        r = client.post("/api/auth/register", json={"email": "a@b.com"})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Auth — login
# ---------------------------------------------------------------------------

class TestLogin:
    def test_success(self, client):
        client.post("/api/auth/register", json={
            "email": "login@ex.com", "username": "loginuser", "password": "password123",
        })
        r = client.post("/api/auth/login", json={
            "email": "login@ex.com", "password": "password123",
        })
        assert r.status_code == 200
        assert "accessToken" in r.get_json()

    def test_wrong_password(self, client):
        r = client.post("/api/auth/login", json={
            "email": "user@test.com", "password": "wrongpassword",
        })
        assert r.status_code == 401

    def test_unknown_email(self, client):
        r = client.post("/api/auth/login", json={
            "email": "nobody@example.com", "password": "password123",
        })
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# Auth — /me
# ---------------------------------------------------------------------------

class TestMe:
    def test_authenticated(self, client, user_headers):
        r = client.get("/api/auth/me", headers=user_headers)
        assert r.status_code == 200
        assert "user" in r.get_json()

    def test_unauthenticated(self, client):
        r = client.get("/api/auth/me")
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_success_anonymous(self, client):
        r = client.post("/api/analyze", json={"text": HINDI, "language": "hindi"})
        assert r.status_code == 200
        d = r.get_json()
        assert d["prediction"] in ("fake", "real")
        assert 0.0 <= d["confidence"] <= 1.0
        assert "scores"       in d
        assert "explanation"  in d
        assert "contentHash"  in d
        assert "blockchain"   in d

    def test_explanation_shape(self, client):
        r = client.post("/api/analyze", json={"text": HINDI, "language": "hindi"})
        exp = r.get_json()["explanation"]
        # Explainer mock returns combined_top_words
        assert "combined_top_words" in exp
        assert isinstance(exp["combined_top_words"], list)

    def test_missing_text(self, client):
        r = client.post("/api/analyze", json={"language": "hindi"})
        assert r.status_code == 400

    def test_unsupported_language(self, client):
        r = client.post("/api/analyze", json={"text": HINDI, "language": "swahili"})
        assert r.status_code == 400
        assert "supported" in r.get_json()

    def test_default_language_is_hindi(self, client):
        r = client.post("/api/analyze", json={"text": HINDI})
        assert r.status_code == 200

    def test_all_four_languages(self, client):
        for lang in ["hindi", "gujarati", "marathi", "telugu"]:
            r = client.post("/api/analyze", json={"text": HINDI, "language": lang})
            assert r.status_code == 200, f"Failed for {lang}"

    def test_authenticated_request(self, client, user_headers):
        r = client.post("/api/analyze", headers=user_headers,
                        json={"text": HINDI, "language": "hindi"})
        assert r.status_code == 200

    def test_content_hash_is_hex64(self, client):
        r = client.post("/api/analyze", json={"text": HINDI, "language": "hindi"})
        h = r.get_json()["contentHash"]
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_text_same_hash(self, client):
        r1 = client.post("/api/analyze", json={"text": HINDI, "language": "hindi"})
        r2 = client.post("/api/analyze", json={"text": HINDI, "language": "hindi"})
        assert r1.get_json()["contentHash"] == r2.get_json()["contentHash"]


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class TestFeedback:
    def test_anonymous_feedback(self, client):
        r = client.post("/api/submit-feedback", json={
            "text": HINDI, "predictedLabel": "fake",
            "correctLabel": "real", "language": "hindi",
        })
        assert r.status_code == 201
        assert "feedback_id" in r.get_json()

    def test_authenticated_feedback_with_notes(self, client, user_headers):
        r = client.post("/api/submit-feedback", headers=user_headers, json={
            "text": HINDI, "predictedLabel": "fake", "correctLabel": "real",
            "language": "hindi", "confidence": 0.94,
            "notes": "This is actually real news.",
        })
        assert r.status_code == 201

    def test_missing_required_fields(self, client):
        r = client.post("/api/submit-feedback", json={"text": HINDI})
        assert r.status_code == 400

    def test_invalid_correct_label(self, client):
        r = client.post("/api/submit-feedback", json={
            "text": HINDI, "predictedLabel": "fake",
            "correctLabel": "banana", "language": "hindi",
        })
        assert r.status_code == 400

    def test_invalid_language(self, client):
        r = client.post("/api/submit-feedback", json={
            "text": HINDI, "predictedLabel": "fake",
            "correctLabel": "real", "language": "klingon",
        })
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------

class TestAdmin:
    def test_history_requires_auth(self, client):
        r = client.get("/api/history")
        assert r.status_code == 401

    def test_history_requires_admin(self, client, user_headers):
        r = client.get("/api/history", headers=user_headers)
        assert r.status_code == 403

    def test_history_as_admin(self, client, admin_headers):
        r = client.get("/api/history", headers=admin_headers)
        assert r.status_code == 200
        assert "articles" in r.get_json()

    def test_history_limit_param(self, client, admin_headers):
        r = client.get("/api/history?limit=5", headers=admin_headers)
        assert r.status_code == 200
        articles = r.get_json()["articles"]
        assert len(articles) <= 5

    def test_feedback_stats_as_admin(self, client, admin_headers):
        r = client.get("/api/feedback/stats", headers=admin_headers)
        assert r.status_code == 200
        d = r.get_json()
        assert "total"    in d
        assert "accuracy" in d

    def test_feedback_stats_as_fact_checker(self, client, checker_headers):
        r = client.get("/api/feedback/stats", headers=checker_headers)
        assert r.status_code == 200

    def test_feedback_stats_denied_for_user(self, client, user_headers):
        r = client.get("/api/feedback/stats", headers=user_headers)
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# Blockchain endpoints (disabled — BLOCKCHAIN=None in test config)
# ---------------------------------------------------------------------------

class TestBlockchainEndpoints:
    def test_register_article_503_when_disabled(self, client, publisher_headers):
        r = client.post("/api/register-article", headers=publisher_headers,
                        json={"text": HINDI, "language": "hindi"})
        assert r.status_code == 503

    def test_add_annotation_503_when_disabled(self, client, checker_headers):
        r = client.post("/api/add-annotation", headers=checker_headers,
                        json={"text": HINDI, "flagType": 1,
                              "ipfsHash": "", "confidence": 90})
        assert r.status_code == 503

    def test_register_article_requires_auth(self, client):
        r = client.post("/api/register-article",
                        json={"text": HINDI, "language": "hindi"})
        assert r.status_code == 401

    def test_add_annotation_requires_auth(self, client):
        r = client.post("/api/add-annotation",
                        json={"text": HINDI, "flagType": 1,
                              "ipfsHash": "", "confidence": 90})
        assert r.status_code == 401

    def test_regular_user_cannot_register_article(self, client, user_headers):
        r = client.post("/api/register-article", headers=user_headers,
                        json={"text": HINDI, "language": "hindi"})
        assert r.status_code == 403