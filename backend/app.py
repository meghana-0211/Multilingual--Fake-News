"""
backend/app.py

Flask application factory.

Run (development):
    cd backend
    python app.py

Or with gunicorn (production):
    gunicorn "app:create_app()" --bind 0.0.0.0:5000 --workers 2
"""

import logging
import os
import sys

from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(env: str = "development", test_config: dict | None = None) -> Flask:
    app = Flask(__name__)

    from config import config
    app.config.from_object(config.get(env, config["default"]))

    if test_config:
        app.config.update(test_config)

    _ensure_db_dir(app)

    from database.models import db
    db.init_app(app)

    CORS(app, origins=app.config["CORS_ORIGINS"])
    JWTManager(app)

    with app.app_context():
        db.create_all()
        logger.info("Database tables ready.")

    _load_ml(app)
    _load_blockchain(app)

    from api.auth import auth_bp
    from api.routes import api_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp)

    _register_error_handlers(app)
    logger.info("App ready.")
    return app


# ---------------------------------------------------------------------------
# ML loader
# ---------------------------------------------------------------------------

def _load_ml(app: Flask) -> None:
    """
    Load order:
      1. FakeNewsDetector  (loads checkpoint — done once)
      2. MultilingualPreprocessor
      3. DetectorAsModelWrapper  (zero-cost shim around the detector)
      4. ExplainabilityEngine(model_wrapper)
    """
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    ml_flow_dir = os.path.join(backend_dir, "ml_flow")
    for p in (backend_dir, ml_flow_dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    model_path = app.config["MODEL_PATH"]

    # 1. Detector --------------------------------------------------------
    try:
        from ml_flow.ml_model import FakeNewsDetector
        detector = FakeNewsDetector.load(
            model_path,
            model_name=app.config["BERT_MODEL"],
        )
        app.config["DETECTOR"] = detector
        logger.info(f"Model loaded: {model_path}")
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        app.config["DETECTOR"] = None

    # 2. Preprocessor ----------------------------------------------------
    try:
        from ml_flow.multilingual_preprocessor import MultilingualPreprocessor
        app.config["PREPROCESSOR"] = MultilingualPreprocessor(
            model_name=app.config["BERT_MODEL"]
        )
        logger.info("Preprocessor ready.")
    except Exception as exc:
        logger.warning(f"Preprocessor not loaded: {exc}")
        app.config["PREPROCESSOR"] = None

    # 3 + 4. Explainer ---------------------------------------------------
    detector = app.config.get("DETECTOR")
    if detector is None:
        logger.warning("Explainer skipped — model not loaded.")
        app.config["EXPLAINER"] = None
        return

    try:
        from ml_flow.ml_model import DetectorAsModelWrapper
        from ml_flow.explainability  import ExplainabilityEngine

        wrapper = DetectorAsModelWrapper(detector)
        app.config["EXPLAINER"] = ExplainabilityEngine(wrapper)
        logger.info("Explainer ready.")
    except Exception as exc:
        logger.warning(f"Explainer not loaded: {exc}")
        app.config["EXPLAINER"] = None


# ---------------------------------------------------------------------------
# Blockchain loader
# ---------------------------------------------------------------------------

def _load_blockchain(app: Flask) -> None:
    if not app.config.get("BLOCKCHAIN_ENABLED", True):
        logger.info("Blockchain disabled via config.")
        app.config["BLOCKCHAIN"] = None
        return

    try:
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        bc_dir      = os.path.join(os.path.dirname(backend_dir), "blockchain")
        if bc_dir not in sys.path:
            sys.path.insert(0, bc_dir)

        from blockchain.web3_client import BlockchainClient
 
        client = BlockchainClient(
            provider_url=app.config["BLOCKCHAIN_URL"],
            publisher_registry_address=app.config["PUBLISHER_REGISTRY_ADDRESS"],
            article_registry_address=app.config["ARTICLE_REGISTRY_ADDRESS"],
            annotation_registry_address=app.config["ANNOTATION_REGISTRY_ADDRESS"],
        )

        private_key = app.config.get("PRIVATE_KEY", "")
        if private_key:
            client.set_account(private_key)

        client.load_contract("PublisherRegistry")
        client.load_contract("ArticleRegistry")
        client.load_contract("AnnotationRegistry")

        app.config["BLOCKCHAIN"] = client
        logger.info("Blockchain client ready.")
    except Exception as exc:
        logger.warning(f"Blockchain unavailable: {exc}")
        app.config["BLOCKCHAIN"] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_db_dir(app: Flask) -> None:
    db_url = app.config.get("SQLALCHEMY_DATABASE_URI", "")
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
        db_dir  = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad request", "detail": str(e)}), 400

    @app.errorhandler(401)
    def unauthorised(e):
        return jsonify({"error": "Unauthorised"}), 401

    @app.errorhandler(403)
    def forbidden(e):
        return jsonify({"error": "Forbidden"}), 403

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def server_error(e):
        logger.exception(e)
        return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = os.environ.get("FLASK_ENV", "development")
    application = create_app(env)
    application.run(
        debug=application.config["DEBUG"],
        host="0.0.0.0",
        port=5001,
    )