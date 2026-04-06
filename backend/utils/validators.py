"""
backend/utils/validators.py

Request-body validation helpers used by routes.
Keeps validation logic out of the route functions.
"""

import re
from typing import Any


class ValidationError(ValueError):
    """Raised when a request field fails validation."""
    pass


# ---------------------------------------------------------------------------
# Field validators
# ---------------------------------------------------------------------------

def require_fields(data: dict, *fields: str) -> None:
    """Raise ValidationError if any required field is missing or blank."""
    for field in fields:
        val = data.get(field)
        if val is None or (isinstance(val, str) and not val.strip()):
            raise ValidationError(f"'{field}' is required")


def validate_email(email: str) -> str:
    """Return normalised email or raise ValidationError."""
    email = email.strip().lower()
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    if not re.match(pattern, email):
        raise ValidationError(f"Invalid email: '{email}'")
    return email


def validate_password(password: str, min_length: int = 8) -> str:
    """Raise ValidationError if password is too short."""
    if len(password) < min_length:
        raise ValidationError(
            f"Password must be at least {min_length} characters"
        )
    return password


def validate_language(language: str,
                      supported: list[str] | None = None) -> str:
    """Return normalised language code or raise ValidationError."""
    supported = supported or ["hindi", "gujarati", "marathi", "telugu"]
    lang = language.strip().lower()
    if lang not in supported:
        raise ValidationError(
            f"Unsupported language '{lang}'. Choose from: {supported}"
        )
    return lang


def validate_label(label: str) -> str:
    """Ensure label is 'fake' or 'real'."""
    label = label.strip().lower()
    if label not in ("fake", "real"):
        raise ValidationError(f"Invalid label '{label}'. Use 'fake' or 'real'.")
    return label


def validate_confidence(value: Any) -> float:
    """Accept 0-1 float or 0-100 int; normalise to 0-1."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValidationError("'confidence' must be a number")

    if 0.0 <= v <= 1.0:
        return round(v, 4)
    if 1.0 < v <= 100.0:
        return round(v / 100.0, 4)

    raise ValidationError("'confidence' must be between 0 and 1 (or 0-100)")


def validate_flag_type(value: Any) -> int:
    """
    Blockchain FlagType enum:
        0=MISLEADING  1=FALSE  2=SATIRE  3=UNVERIFIED  4=CORRECT
    """
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValidationError("'flagType' must be an integer 0-4")

    if v not in range(5):
        raise ValidationError("'flagType' must be 0-4")
    return v