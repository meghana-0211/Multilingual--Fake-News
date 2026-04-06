"""
backend/api/auth.py

Authentication blueprint — register, login, /me.
JWT tokens issued by flask-jwt-extended.
Role-based access decorators also live here so routes can import them.
"""

import logging
from datetime import datetime, timezone
from functools import wraps

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
    get_jwt_identity,
)

from backend.database.crud   import (create_user, get_user_by_email,
                              get_user_by_id, update_last_login)
from backend.database.models import UserRole

logger = logging.getLogger(__name__)
auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


# ---------------------------------------------------------------------------
# Role decorators
# ---------------------------------------------------------------------------

def roles_required(*roles: str):
    """
    Decorator that enforces one or more allowed roles.
    Must be used *after* @jwt_required().

    Usage:
        @auth_bp.route("/admin-only")
        @jwt_required()
        @roles_required("admin")
        def admin_only(): ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user_id = get_jwt_identity()
            user    = get_user_by_id(int(user_id))
            if user is None or user.role.value not in roles:
                return jsonify({"error": "Insufficient permissions"}), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def get_current_user():
    """Return the User object for the current JWT identity."""
    user_id = get_jwt_identity()
    return get_user_by_id(int(user_id)) if user_id else None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@auth_bp.route("/register", methods=["POST"])
def register():
    """
    POST /api/auth/register
    Body: { email, username, password }

    Test:
        curl -X POST http://localhost:5000/api/auth/register \
          -H "Content-Type: application/json" \
          -d '{"email":"a@b.com","username":"alice","password":"secret123"}'
    """
    data = request.get_json(silent=True) or {}

    email    = (data.get("email")    or "").strip().lower()
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    if not email or not username or not password:
        return jsonify({"error": "email, username and password are required"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    try:
        user = create_user(email, username, password)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    token = create_access_token(identity=str(user.id))
    logger.info(f"Registered: {email}")
    return jsonify({
        "accessToken": token,
        "user":        user.to_dict(),
    }), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    """
    POST /api/auth/login
    Body: { email, password }

    Test:
        curl -X POST http://localhost:5000/api/auth/login \
          -H "Content-Type: application/json" \
          -d '{"email":"a@b.com","password":"secret123"}'
    """
    data = request.get_json(silent=True) or {}

    email    = (data.get("email")    or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    user = get_user_by_email(email)
    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid credentials"}), 401
    if not user.is_active:
        return jsonify({"error": "Account deactivated"}), 403

    update_last_login(user)
    token = create_access_token(identity=str(user.id))
    logger.info(f"Login: {email}")
    return jsonify({
        "accessToken": token,
        "user":        user.to_dict(),
    })


@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def me():
    """
    GET /api/auth/me  — returns the current user's profile.

    Test:
        curl http://localhost:5000/api/auth/me \
          -H "Authorization: Bearer <token>"
    """
    user = get_current_user()
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"user": user.to_dict()})