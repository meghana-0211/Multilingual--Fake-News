"""
backend/database/crud.py

All database read/write operations.
Routes import from here — never touch db.session directly in routes.
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

from .models import db, User, UserRole, ArticleLog, Feedback, Language, PredictionLabel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash(text: str) -> str:
    """SHA-256 hex digest of text (same logic as blockchain/web3_client.py)."""
    normalised = " ".join(text.strip().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _to_language(lang: str) -> Language:
    try:
        return Language(lang.lower())
    except ValueError:
        raise ValueError(
            f"Unsupported language '{lang}'. "
            f"Choose from: {[l.value for l in Language]}"
        )


def _to_label(label: str) -> PredictionLabel:
    try:
        return PredictionLabel(label.lower())
    except ValueError:
        raise ValueError(f"Invalid label '{label}'. Use 'fake' or 'real'.")


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------

def create_user(email: str, username: str, password: str,
                role: str = "user") -> User:
    """Create and persist a new user. Raises ValueError on duplicate."""
    if User.query.filter_by(email=email).first():
        raise ValueError(f"Email already registered: {email}")
    if User.query.filter_by(username=username).first():
        raise ValueError(f"Username taken: {username}")

    user = User(
        email=email,
        username=username,
        role=UserRole(role),
    )
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    logger.info(f"Created user: {email}")
    return user


def get_user_by_email(email: str) -> Optional[User]:
    return User.query.filter_by(email=email).first()


def get_user_by_id(user_id: int) -> Optional[User]:
    return db.session.get(User, user_id)


def update_last_login(user: User) -> None:
    user.last_login = datetime.now(timezone.utc)
    db.session.commit()


# ---------------------------------------------------------------------------
# ArticleLog CRUD
# ---------------------------------------------------------------------------

def log_article(
    text: str,
    language: str,
    prediction: str,
    confidence: float,
    fake_probability: float,
    real_probability: float,
    blockchain_hash: Optional[str] = None,
    blockchain_verified: bool = False,
    user_id: Optional[int] = None,
) -> ArticleLog:
    """
    Persist one analysis result.
    Returns the saved ArticleLog row.
    """
    content_hash = _hash(text)

    log = ArticleLog(
        content_hash=content_hash,
        text_snippet=text[:500],
        language=_to_language(language),
        prediction=_to_label(prediction),
        confidence=confidence,
        fake_probability=fake_probability,
        real_probability=real_probability,
        blockchain_hash=blockchain_hash,
        blockchain_verified=blockchain_verified,
        user_id=user_id,
    )
    db.session.add(log)
    db.session.commit()
    return log


def get_article_by_hash(content_hash: str) -> Optional[ArticleLog]:
    """Return the most recent analysis for a given content hash."""
    return (ArticleLog.query
            .filter_by(content_hash=content_hash)
            .order_by(ArticleLog.created_at.desc())
            .first())


def get_recent_articles(limit: int = 50) -> list[ArticleLog]:
    return (ArticleLog.query
            .order_by(ArticleLog.created_at.desc())
            .limit(limit)
            .all())


# ---------------------------------------------------------------------------
# Feedback CRUD
# ---------------------------------------------------------------------------

def create_feedback(
    article_text: str,
    language: str,
    predicted_label: str,
    correct_label: str,
    confidence: Optional[float] = None,
    notes: Optional[str] = None,
    user_id: Optional[int] = None,
) -> Feedback:
    """
    Persist a user correction.
    Links to an existing ArticleLog row if one exists for the same hash.
    """
    content_hash = _hash(article_text)
    article = get_article_by_hash(content_hash)

    fb = Feedback(
        article_id=article.id if article else None,
        user_id=user_id,
        article_text=article_text,
        content_hash=content_hash,
        language=_to_language(language),
        predicted_label=_to_label(predicted_label),
        correct_label=_to_label(correct_label),
        confidence=confidence,
        notes=notes,
    )
    db.session.add(fb)
    db.session.commit()
    logger.info(f"Feedback saved: predicted={predicted_label} correct={correct_label}")
    return fb


def get_unverified_feedback(limit: int = 100) -> list[Feedback]:
    """Return feedback not yet reviewed by a fact-checker."""
    return (Feedback.query
            .filter_by(is_verified=False)
            .order_by(Feedback.created_at.desc())
            .limit(limit)
            .all())


def mark_feedback_verified(feedback_id: int) -> Optional[Feedback]:
    fb = db.session.get(Feedback, feedback_id)
    if fb:
        fb.is_verified = True
        db.session.commit()
    return fb


def get_feedback_stats() -> dict:
    """Quick stats for the admin/model-improvement pipeline."""
    total   = Feedback.query.count()
    correct = Feedback.query.filter(
        Feedback.predicted_label == Feedback.correct_label
    ).count()
    wrong   = total - correct
    return {
        "total":    total,
        "correct":  correct,
        "wrong":    wrong,
        "accuracy": round(correct / total, 4) if total else 0.0,
    }