"""
backend/api/routes.py

Main API blueprint.

Endpoints
---------
GET  /api/health               – liveness probe
POST /api/analyze              – fake-news detection + explanation
POST /api/submit-feedback      – user correction (JWT optional)
POST /api/register-article     – blockchain registration (publisher+)
POST /api/add-annotation       – fact-check flag (fact_checker+)
GET  /api/history              – recent analyses (admin)
GET  /api/feedback/stats       – feedback stats (admin / fact_checker)
"""

import hashlib
import logging

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity, verify_jwt_in_request

from backend.database.crud import (
    log_article, create_feedback,
    get_recent_articles, get_feedback_stats,
)
from api.auth import roles_required

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__, url_prefix="/api")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    normalised = " ".join(text.strip().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _detector():
    return current_app.config.get("DETECTOR")


def _explainer():
    return current_app.config.get("EXPLAINER")


def _blockchain():
    return current_app.config.get("BLOCKCHAIN")


def _optional_user_id():
    """Extract JWT user_id if a valid token is present; return None otherwise."""
    try:
        verify_jwt_in_request(optional=True)
        raw = get_jwt_identity()
        return int(raw) if raw else None
    except Exception:
        return None


def _serialize_explanation(raw: dict) -> dict:
    """
    Convert ExplainabilityEngine.explain_comprehensive() output to a
    JSON-safe dict.  Strips numpy arrays and non-serialisable objects.
    """
    if not raw:
        return {}

    def _safe_features(pairs):
        return [{"word": str(w), "score": float(s)} for w, s in (pairs or [])]

    lime = raw.get("lime_explanation", {})
    attn = raw.get("attention_explanation", {})

    return {
        "prediction": {
            "label":      raw.get("prediction", {}).get("label"),
            "confidence": raw.get("prediction", {}).get("confidence"),
        },
        "combined_top_words":  _safe_features(raw.get("combined_top_words", [])),
        "lime_features":       _safe_features(lime.get("feature_importance", [])),
        "attention_top_tokens": _safe_features(attn.get("top_tokens", [])),
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@api_bp.route("/health", methods=["GET"])
def health():
    """
    GET /api/health

    Test:
        curl http://localhost:5000/api/health
    """
    return jsonify({
        "status":        "ok",
        "model_loaded":  _detector() is not None,
        "blockchain":    _blockchain() is not None,
        "explainer":     _explainer() is not None,
    })


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

@api_bp.route("/analyze", methods=["POST"])
def analyze_article():
    """
    POST /api/analyze
    Body: { "text": "...", "language": "hindi" }

    Test (no auth required):
        curl -X POST http://localhost:5000/api/analyze \\
          -H "Content-Type: application/json" \\
          -d '{"text":"कोरोना वायरस से बचने के लिए गर्म पानी पीना काफी है।","language":"hindi"}'

    Response shape:
        {
          "prediction":  "fake" | "real",
          "label":       "Fake" | "Real",
          "confidence":  0.94,
          "scores":      {"fake": 0.94, "real": 0.06},
          "explanation": {
              "combined_top_words":   [{"word": "...", "score": ...}, ...],
              "lime_features":        [...],
              "attention_top_tokens": [...]
          },
          "contentHash": "a3b2c1...",
          "blockchain":  { "verified": false, ... }
        }
    """
    data     = request.get_json(silent=True) or {}
    text     = (data.get("text")     or "").strip()
    language = (data.get("language") or "hindi").strip().lower()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    supported = current_app.config.get(
        "SUPPORTED_LANGUAGES", ["hindi", "gujarati", "marathi", "telugu"]
    )
    if language not in supported:
        return jsonify({"error": f"Unsupported language '{language}'",
                        "supported": supported}), 400

    detector = _detector()
    if detector is None:
        return jsonify({"error": "Model not available"}), 503

    # --- ML inference ---
    result = detector.predict(text)

    # --- Explanation (non-fatal) ---
    explanation = {}
    explainer   = _explainer()
    if explainer:
        try:
            raw_exp     = explainer.explain_comprehensive(text)
            explanation = _serialize_explanation(raw_exp)
        except Exception as exc:
            logger.warning(f"Explainer failed: {exc}")
            explanation = {"error": "Explanation unavailable"}

    # --- Blockchain lookup (non-fatal) ---
    content_hash    = _sha256(text)
    blockchain_info = {"verified": False, "publisher": None,
                       "timestamp": None, "annotations": []}
    bc = _blockchain()
    if bc:
        try:
            h             = bytes.fromhex(content_hash)
            verification  = bc.verify_article(h)
            annotations   = bc.get_annotations(h)
            blockchain_info = {
                "verified":    verification["exists"],
                "publisher":   verification.get("publisher"),
                "timestamp":   verification.get("timestamp"),
                "annotations": annotations,
            }
        except Exception as exc:
            logger.warning(f"Blockchain lookup failed: {exc}")

    # --- Persist ---
    user_id = _optional_user_id()
    try:
        log_article(
            text=text,
            language=language,
            prediction=result["prediction"],
            confidence=result["confidence"],
            fake_probability=result["probabilities"]["fake"],
            real_probability=result["probabilities"]["real"],
            blockchain_verified=blockchain_info["verified"],
            user_id=user_id,
        )
    except Exception as exc:
        logger.error(f"DB log failed: {exc}")

    return jsonify({
        "prediction":  result["prediction"],
        "label":       result["label"],
        "confidence":  result["confidence"],
        "scores": {
            "fake": result["probabilities"]["fake"],
            "real": result["probabilities"]["real"],
        },
        "explanation":  explanation,
        "contentHash":  content_hash,
        "blockchain":   blockchain_info,
    })


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

@api_bp.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    """
    POST /api/submit-feedback
    Body: {
        "text":           "...",
        "predictedLabel": "fake",
        "correctLabel":   "real",
        "language":       "hindi",
        "confidence":     0.94,    // optional
        "notes":          "..."    // optional
    }
    JWT is optional — anonymous feedback accepted.

    Test:
        curl -X POST http://localhost:5000/api/submit-feedback \\
          -H "Content-Type: application/json" \\
          -d '{"text":"...","predictedLabel":"fake","correctLabel":"real","language":"hindi"}'
    """
    data = request.get_json(silent=True) or {}

    text            = (data.get("text")           or "").strip()
    predicted_label = (data.get("predictedLabel") or "").strip().lower()
    correct_label   = (data.get("correctLabel")   or "").strip().lower()
    language        = (data.get("language")       or "hindi").strip().lower()
    confidence      = data.get("confidence")
    notes           = data.get("notes")

    if not text or not predicted_label or not correct_label:
        return jsonify({"error": "text, predictedLabel and correctLabel required"}), 400

    try:
        fb = create_feedback(
            article_text=text,
            language=language,
            predicted_label=predicted_label,
            correct_label=correct_label,
            confidence=confidence,
            notes=notes,
            user_id=_optional_user_id(),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"message": "Feedback submitted successfully",
                    "feedback_id": fb.id}), 201


# ---------------------------------------------------------------------------
# Blockchain: register article (publisher / admin)
# ---------------------------------------------------------------------------

@api_bp.route("/register-article", methods=["POST"])
@jwt_required()
@roles_required("publisher", "admin")
def register_article():
    """
    POST /api/register-article  — publisher / admin only.
    Body: { "text": "...", "language": "hindi" }

    Test:
        curl -X POST http://localhost:5000/api/register-article \\
          -H "Content-Type: application/json" \\
          -H "Authorization: Bearer <token>" \\
          -d '{"text":"Article text...","language":"hindi"}'
    """
    bc = _blockchain()
    if not bc:
        return jsonify({"error": "Blockchain integration is disabled"}), 503

    data     = request.get_json(silent=True) or {}
    text     = (data.get("text")     or "").strip()
    language = (data.get("language") or "hindi").strip().lower()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        receipt = bc.register_article(bytes.fromhex(_sha256(text)), language)
        tx_hash = receipt["transactionHash"]
        return jsonify({
            "success":         True,
            "transactionHash": tx_hash.hex() if hasattr(tx_hash, "hex") else str(tx_hash),
            "contentHash":     _sha256(text),
        })
    except Exception as exc:
        logger.error(f"register_article failed: {exc}")
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Blockchain: add annotation (fact_checker / admin)
# ---------------------------------------------------------------------------

@api_bp.route("/add-annotation", methods=["POST"])
@jwt_required()
@roles_required("fact_checker", "admin")
def add_annotation():
    """
    POST /api/add-annotation  — fact_checker / admin only.
    Body: {
        "text":       "...",
        "flagType":   1,        // 0=MISLEADING 1=FALSE 2=SATIRE 3=UNVERIFIED 4=CORRECT
        "ipfsHash":   "Qm...",  // optional
        "confidence": 90        // 0-100
    }

    Test:
        curl -X POST http://localhost:5000/api/add-annotation \\
          -H "Content-Type: application/json" \\
          -H "Authorization: Bearer <token>" \\
          -d '{"text":"...","flagType":1,"ipfsHash":"","confidence":90}'
    """
    bc = _blockchain()
    if not bc:
        return jsonify({"error": "Blockchain integration is disabled"}), 503

    data       = request.get_json(silent=True) or {}
    text       = (data.get("text") or "").strip()
    flag_type  = data.get("flagType",   1)
    ipfs_hash  = data.get("ipfsHash",   "")
    confidence = data.get("confidence", 80)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        receipt = bc.add_annotation(
            bytes.fromhex(_sha256(text)), flag_type, ipfs_hash, confidence
        )
        tx_hash = receipt["transactionHash"]
        return jsonify({
            "success":         True,
            "transactionHash": tx_hash.hex() if hasattr(tx_hash, "hex") else str(tx_hash),
        })
    except Exception as exc:
        logger.error(f"add_annotation failed: {exc}")
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

@api_bp.route("/history", methods=["GET"])
@jwt_required()
@roles_required("admin")
def history():
    """
    GET /api/history?limit=50  — admin only.

    Test:
        curl "http://localhost:5000/api/history?limit=10" \\
          -H "Authorization: Bearer <admin_token>"
    """
    limit   = min(int(request.args.get("limit", 50)), 200)
    records = get_recent_articles(limit)
    return jsonify({"articles": [r.to_dict() for r in records]})


@api_bp.route("/feedback/stats", methods=["GET"])
@jwt_required()
@roles_required("admin", "fact_checker")
def feedback_stats():
    """
    GET /api/feedback/stats

    Test:
        curl http://localhost:5000/api/feedback/stats \\
          -H "Authorization: Bearer <admin_token>"
    """
    return jsonify(get_feedback_stats())