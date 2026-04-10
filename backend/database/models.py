"""
backend/database/models.py

SQLAlchemy ORM models.
SQLite by default (DATABASE_URL in config); swap for Postgres in prod
by just changing the URL — no code changes needed.
"""

import enum
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class UserRole(str, enum.Enum):
    USER         = "user"
    FACT_CHECKER = "fact_checker"
    PUBLISHER    = "publisher"
    ADMIN        = "admin"


class Language(str, enum.Enum):
    HINDI    = "hindi"
    GUJARATI = "gujarati"
    MARATHI  = "marathi"
    TELUGU   = "telugu"


class PredictionLabel(str, enum.Enum):
    FAKE = "fake"
    REAL = "real"


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------

class User(db.Model):
    __tablename__ = "users"

    id         = db.Column(db.Integer, primary_key=True)
    email      = db.Column(db.String(255), unique=True, nullable=False, index=True)
    username   = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role       = db.Column(db.Enum(UserRole), nullable=False,
                           default=UserRole.USER)
    is_active  = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False,
                           default=lambda: datetime.now(timezone.utc))
    last_login = db.Column(db.DateTime, nullable=True)

    # relationships
    feedbacks  = db.relationship("Feedback",   back_populates="user",
                                 lazy="dynamic")
    articles   = db.relationship("ArticleLog", back_populates="user",
                                 lazy="dynamic")

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "email":      self.email,
            "username":   self.username,
            "role":       self.role.value,
            "is_active":  self.is_active,
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self) -> str:
        return f"<User {self.email} ({self.role.value})>"


# ---------------------------------------------------------------------------
# ArticleLog — every article analysed, one row per request
# ---------------------------------------------------------------------------

class ArticleLog(db.Model):
    __tablename__ = "article_logs"

    id               = db.Column(db.Integer, primary_key=True)
    content_hash     = db.Column(db.String(64), nullable=False, index=True)
    text_snippet     = db.Column(db.Text, nullable=False)   # first 500 chars
    language         = db.Column(db.Enum(Language), nullable=False)
    prediction       = db.Column(db.Enum(PredictionLabel), nullable=False)
    confidence       = db.Column(db.Float, nullable=False)
    fake_probability = db.Column(db.Float, nullable=False)
    real_probability = db.Column(db.Float, nullable=False)
    blockchain_hash  = db.Column(db.String(66), nullable=True)   # tx hash
    blockchain_verified = db.Column(db.Boolean, default=False)
    user_id          = db.Column(db.Integer, db.ForeignKey("users.id"),
                                 nullable=True)
    created_at       = db.Column(db.DateTime, nullable=False,
                                 default=lambda: datetime.now(timezone.utc))

    user      = db.relationship("User",     back_populates="articles")
    feedbacks = db.relationship("Feedback", back_populates="article",
                                lazy="dynamic")

    def to_dict(self) -> dict:
        return {
            "id":                  self.id,
            "content_hash":        self.content_hash,
            "language":            self.language.value,
            "prediction":          self.prediction.value,
            "confidence":          self.confidence,
            "fake_probability":    self.fake_probability,
            "real_probability":    self.real_probability,
            "blockchain_verified": self.blockchain_verified,
            "created_at":          self.created_at.isoformat(),
        }

    def __repr__(self) -> str:
        return (f"<ArticleLog {self.content_hash[:8]}… "
                f"{self.prediction.value} ({self.confidence:.2f})>")


# ---------------------------------------------------------------------------
# Feedback — user correction / annotation
# ---------------------------------------------------------------------------

class Feedback(db.Model):
    __tablename__ = "feedbacks"

    id              = db.Column(db.Integer, primary_key=True)
    article_id      = db.Column(db.Integer, db.ForeignKey("article_logs.id"),
                                nullable=True)
    user_id         = db.Column(db.Integer, db.ForeignKey("users.id"),
                                nullable=True)
    article_text    = db.Column(db.Text, nullable=False)
    content_hash    = db.Column(db.String(64), nullable=False, index=True)
    language        = db.Column(db.Enum(Language), nullable=False)
    predicted_label = db.Column(db.Enum(PredictionLabel), nullable=False)
    correct_label   = db.Column(db.Enum(PredictionLabel), nullable=False)
    confidence      = db.Column(db.Float, nullable=True)
    notes           = db.Column(db.Text, nullable=True)
    is_verified     = db.Column(db.Boolean, default=False)  # fact-checker reviewed
    created_at      = db.Column(db.DateTime, nullable=False,
                                default=lambda: datetime.now(timezone.utc))

    user    = db.relationship("User",       back_populates="feedbacks")
    article = db.relationship("ArticleLog", back_populates="feedbacks")

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "content_hash":    self.content_hash,
            "language":        self.language.value,
            "predicted_label": self.predicted_label.value,
            "correct_label":   self.correct_label.value,
            "confidence":      self.confidence,
            "is_verified":     self.is_verified,
            "created_at":      self.created_at.isoformat(),
        }

    def __repr__(self) -> str:
        return (f"<Feedback predicted={self.predicted_label.value} "
                f"correct={self.correct_label.value}>")