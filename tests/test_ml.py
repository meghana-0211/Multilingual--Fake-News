"""
backend/tests/test_ml.py

Unit tests for:
  - FakeNewsDetector (mocked model — no checkpoint needed)
  - DetectorAsModelWrapper  (interface contract tests)
  - database CRUD operations

Run:
    cd backend
    pytest tests/test_ml.py -v
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model_output(batch_size: int = 1, fake_logit: float = 3.0):
    """Returns a dict matching IndicBERTBiLSTMEnsemble.forward() output."""
    logits = torch.tensor([[0.1, fake_logit]] * batch_size)
    return {
        "logits":            logits,
        "bert_logits":       logits,
        "lstm_logits":       logits,
        "attention_weights": torch.zeros(batch_size, 10, 10),
    }


# ---------------------------------------------------------------------------
# FakeNewsDetector
# ---------------------------------------------------------------------------

class TestFakeNewsDetector:
    @pytest.fixture
    def detector(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

        from backend.ml_flow.ml_model import FakeNewsDetector, IndicBERTBiLSTMEnsemble
        from transformers  import AutoTokenizer

        tokenizer    = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
        device       = torch.device("cpu")
        mock_model   = MagicMock(spec=IndicBERTBiLSTMEnsemble)
        mock_model.return_value = _make_mock_model_output(1)
        mock_model.eval.return_value = mock_model

        return FakeNewsDetector(mock_model, tokenizer, device)

    def test_predict_keys(self, detector):
        r = detector.predict("यह एक परीक्षण है।")
        for key in ("prediction", "label", "confidence", "probabilities"):
            assert key in r

    def test_predict_is_fake(self, detector):
        r = detector.predict("test text")
        assert r["prediction"] == "fake"
        assert r["label"]      == "Fake"

    def test_predict_real_when_real_logit_higher(self, detector):
        detector.model.return_value = _make_mock_model_output(1, fake_logit=-3.0)
        r = detector.predict("test text")
        assert r["prediction"] == "real"

    def test_probabilities_sum_to_one(self, detector):
        r = detector.predict("test")
        total = r["probabilities"]["real"] + r["probabilities"]["fake"]
        assert abs(total - 1.0) < 1e-4

    def test_confidence_matches_max_prob(self, detector):
        r = detector.predict("test")
        max_prob = max(r["probabilities"].values())
        assert abs(r["confidence"] - max_prob) < 1e-4

    def test_batch_proba_shape(self, detector):
        detector.model.return_value = _make_mock_model_output(3)
        proba = detector.predict_batch_proba(["a", "b", "c"])
        assert proba.shape == (3, 2)

    def test_batch_rows_sum_to_one(self, detector):
        detector.model.return_value = _make_mock_model_output(2)
        proba = detector.predict_batch_proba(["a", "b"])
        for row in proba:
            assert abs(row.sum() - 1.0) < 1e-4

    def test_empty_batch(self, detector):
        result = detector.predict_batch_proba([])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# DetectorAsModelWrapper — interface contract
# ---------------------------------------------------------------------------

class TestDetectorAsModelWrapper:
    @pytest.fixture
    def wrapper(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

        from backend.ml_flow.ml_model import (
            FakeNewsDetector, DetectorAsModelWrapper, IndicBERTBiLSTMEnsemble
        )
        from transformers import AutoTokenizer

        tokenizer  = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
        device     = torch.device("cpu")
        mock_model = MagicMock(spec=IndicBERTBiLSTMEnsemble)
        mock_model.return_value = _make_mock_model_output(1)
        mock_model.eval.return_value = mock_model

        det = FakeNewsDetector(mock_model, tokenizer, device)
        return DetectorAsModelWrapper(det)

    def test_predict_single_returns_int_prediction(self, wrapper):
        """ExplainabilityEngine expects prediction as int 0/1."""
        r = wrapper.predict_single("test text")
        assert r["prediction"] in (0, 1)

    def test_predict_single_fake_is_1(self, wrapper):
        r = wrapper.predict_single("test text")
        assert r["prediction"] == 1        # fake

    def test_predict_batch_returns_ndarray(self, wrapper):
        wrapper._det.model.return_value = _make_mock_model_output(2)
        proba = wrapper.predict_batch(["a", "b"])
        assert isinstance(proba, np.ndarray)
        assert proba.shape == (2, 2)

    def test_get_attention_weights_keys(self, wrapper):
        attn = wrapper.get_attention_weights("यह एक परीक्षण है।")
        assert "tokens"            in attn
        assert "attention_weights" in attn
        assert "prediction"        in attn

    def test_get_attention_weights_matrix_is_square(self, wrapper):
        attn   = wrapper.get_attention_weights("test")
        matrix = attn["attention_weights"]
        assert matrix.ndim == 2
        assert matrix.shape[0] == matrix.shape[1]

    def test_exposes_tokenizer(self, wrapper):
        assert wrapper.tokenizer is not None

    def test_exposes_device(self, wrapper):
        assert wrapper.device is not None


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

class TestCRUD:
    @pytest.fixture(scope="class")
    def ctx(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

        from backend.app import create_app
        from backend.database.models import db

        app = create_app("development", test_config={
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "DETECTOR": None,
            "EXPLAINER": None,
            "PREPROCESSOR": None,
            "BLOCKCHAIN": None,
            "BLOCKCHAIN_ENABLED": False,
        })

        with app.app_context():
            db.drop_all()      # 🔥 important
            db.create_all()

        yield app

        with app.app_context():
            db.session.remove()
            db.drop_all()

    def test_create_and_fetch_user(self, ctx):
        with ctx.app_context():
            from backend.database.crud import create_user, get_user_by_email
            u = create_user("crud@test.com", "cruduser", "password123")
            assert u.id is not None
            assert get_user_by_email("crud@test.com").username == "cruduser"

    def test_duplicate_email_raises(self, ctx):
        with ctx.app_context():
            from backend.database.crud import create_user
            create_user("dup@crud.com", "dupA", "password123")
            with pytest.raises(ValueError, match="already registered"):
                create_user("dup@crud.com", "dupB", "password123")

    def test_password_check(self, ctx):
        with ctx.app_context():
            from backend.database.crud import create_user, get_user_by_email
            create_user("pw@crud.com", "pwuser", "mypassword")
            u = get_user_by_email("pw@crud.com")
            assert     u.check_password("mypassword")
            assert not u.check_password("wrongpassword")

    def test_log_article(self, ctx):
        with ctx.app_context():
            import hashlib
            from backend.database.crud import log_article, get_article_by_hash
            text = "unique article text for log test"
            log_article(
                text=text, language="hindi", prediction="fake",
                confidence=0.9, fake_probability=0.9, real_probability=0.1,
            )
            h = hashlib.sha256(text.encode()).hexdigest()
            r = get_article_by_hash(h)
            assert r is not None
            assert r.prediction.value == "fake"

    def test_create_feedback(self, ctx):
        with ctx.app_context():
            from backend.database.crud import create_feedback
            fb = create_feedback(
                article_text="some article text",
                language="hindi",
                predicted_label="fake",
                correct_label="real",
            )
            assert fb.id is not None
            assert fb.correct_label.value == "real"

    def test_feedback_invalid_language(self, ctx):
        with ctx.app_context():
            from backend.database.crud import create_feedback
            with pytest.raises(ValueError, match="Unsupported language"):
                create_feedback("text", "martian", "fake", "real")

    def test_feedback_invalid_label(self, ctx):
        with ctx.app_context():
            from backend.database.crud import create_feedback
            with pytest.raises(ValueError, match="Invalid label"):
                create_feedback("text", "hindi", "fake", "banana")

    def test_feedback_stats(self, ctx):
        with ctx.app_context():
            from backend.database.crud import get_feedback_stats
            s = get_feedback_stats()
            assert "total"    in s
            assert "accuracy" in s
            assert 0.0 <= s["accuracy"] <= 1.0

    def test_get_recent_articles(self, ctx):
        with ctx.app_context():
            from backend.database.crud import get_recent_articles
            records = get_recent_articles(limit=10)
            assert isinstance(records, list)


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

class TestUtils:
    def test_hash_text_is_64_hex(self):
        from backend.utils.crypto import hash_text
        h = hash_text("hello world")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_text_normalises_whitespace(self):
        from backend.utils.crypto import hash_text
        assert hash_text("hello  world") == hash_text("hello world")
        assert hash_text("  hello world  ") == hash_text("hello world")

    def test_hash_text_bytes(self):
        from backend.utils.crypto import hash_text, hash_text_bytes
        assert hash_text_bytes("hello") == bytes.fromhex(hash_text("hello"))

    def test_validate_email_valid(self):
        from backend.utils.validators import validate_email
        assert validate_email("Test@Example.COM") == "test@example.com"

    def test_validate_email_invalid(self):
        from backend.utils.validators import validate_email, ValidationError
        with pytest.raises(ValidationError):
            validate_email("not-an-email")

    def test_validate_password_ok(self):
        from backend.utils.validators import validate_password
        assert validate_password("longenough") == "longenough"

    def test_validate_password_too_short(self):
        from backend.utils.validators import validate_password, ValidationError
        with pytest.raises(ValidationError):
            validate_password("short")

    def test_validate_language_ok(self):
        from backend.utils.validators import validate_language
        assert validate_language("Hindi") == "hindi"

    def test_validate_language_bad(self):
        from backend.utils.validators import validate_language, ValidationError
        with pytest.raises(ValidationError):
            validate_language("klingon")

    def test_validate_confidence_float(self):
        from backend.utils.validators import validate_confidence
        assert validate_confidence(0.94) == 0.94

    def test_validate_confidence_int_scale(self):
        from backend.utils.validators import validate_confidence
        assert validate_confidence(94) == pytest.approx(0.94, abs=1e-2)

    def test_validate_flag_type_valid(self):
        from backend.utils.validators import validate_flag_type
        for i in range(5):
            assert validate_flag_type(i) == i

    def test_validate_flag_type_out_of_range(self):
        from backend.utils.validators import validate_flag_type, ValidationError
        with pytest.raises(ValidationError):
            validate_flag_type(5)