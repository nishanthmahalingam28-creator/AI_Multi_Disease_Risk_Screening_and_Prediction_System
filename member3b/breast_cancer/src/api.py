"""
Breast Cancer Prediction Module - Flask REST API Layer.

Exposes the Breast Cancer Prediction Service through standardized REST endpoints.
Delegates all prediction, validation, and metadata logic to BreastCancerPredictionService.

Medical Safety Disclaimer:
This Breast Cancer prediction API is a machine-learning screening/risk prediction
component based on the trained dataset model. It is NOT a medical diagnosis and must
not replace evaluation by a qualified healthcare professional.
"""

from typing import Optional, Dict, Any, Tuple
from flask import Flask, request, jsonify, Response

from .service import BreastCancerPredictionService, DISEASE_KEY


def create_app(
    test_config: Optional[Dict[str, Any]] = None,
    service: Optional[BreastCancerPredictionService] = None,
) -> Flask:
    """
    Application factory for the Breast Cancer Flask REST API.

    Args:
        test_config: Optional configuration dictionary for testing.
        service: Optional pre-configured BreastCancerPredictionService instance.

    Returns:
        Flask: Configured Flask application instance.
    """
    app = Flask(__name__)

    # Default application settings
    app.config["JSON_SORT_KEYS"] = False
    if test_config is not None:
        app.config.update(test_config)

    # Enable CORS for local frontend communication
    try:
        from flask_cors import CORS
        CORS(app, resources={r"/api/*": {"origins": "*"}})
    except ImportError:
        @app.after_request
        def add_cors_headers(response: Response) -> Response:
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            return response

    # Instantiate prediction service if not injected
    bc_service = service if service is not None else BreastCancerPredictionService()

    # -------------------------------------------------------------
    # REST API Routes
    # -------------------------------------------------------------

    @app.route("/api/breast-cancer/health", methods=["GET"])
    def health_check() -> Response:
        """
        Health and capability check endpoint.
        Returns 200 OK if model and preprocessor are ready, 503 if unavailable.
        """
        health_status = bc_service.health_check()
        http_code = 200 if health_status.get("status") == "ready" else 503
        return jsonify(health_status), http_code

    @app.route("/api/breast-cancer/model-info", methods=["GET"])
    def model_info() -> Response:
        """
        Model metadata endpoint.
        Returns active model identity, threshold, feature count, and class mapping.
        """
        info = bc_service.get_model_info()
        return jsonify(info), 200

    def _extract_json_payload(req: request) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[Response, int]]]:
        """
        Safely extracts and validates JSON payload from request, detecting
        duplicate keys, invalid formats, and non-object bodies.
        """
        if not req.is_json:
            return None, (jsonify({
                "status": "error",
                "disease": DISEASE_KEY,
                "error_type": "invalid_request",
                "message": "Request body must contain valid JSON.",
            }), 400)

        raw_text = req.get_data(as_text=True)
        if not raw_text or not raw_text.strip():
            return None, (jsonify({
                "status": "error",
                "disease": DISEASE_KEY,
                "error_type": "invalid_request",
                "message": "Request body must contain valid JSON.",
            }), 400)

        def _detect_duplicates(pairs):
            d = {}
            for k, v in pairs:
                if k in d:
                    raise ValueError(f"Duplicate feature key encountered: '{k}'")
                d[k] = v
            return d

        try:
            import json as _json
            parsed = _json.loads(raw_text, object_pairs_hook=_detect_duplicates)
        except ValueError as ve:
            return None, (jsonify({
                "status": "error",
                "disease": DISEASE_KEY,
                "error_type": "validation_error",
                "message": str(ve),
            }), 400)
        except Exception:
            return None, (jsonify({
                "status": "error",
                "disease": DISEASE_KEY,
                "error_type": "invalid_request",
                "message": "Request body must contain valid JSON.",
            }), 400)

        if not isinstance(parsed, dict):
            return None, (jsonify({
                "status": "error",
                "disease": DISEASE_KEY,
                "error_type": "invalid_request",
                "message": "Request body must contain valid JSON object.",
            }), 400)

        return parsed, None

    @app.route("/api/breast-cancer/predict/user", methods=["POST"])
    def predict_user_endpoint() -> Response:
        """
        User-facing screening prediction endpoint.
        Accepts flat or grouped 30-feature JSON payload and returns risk percentage.
        """
        data, err_resp = _extract_json_payload(request)
        if err_resp is not None:
            resp, code = err_resp
            return resp, code

        result = bc_service.predict_user(data)
        http_code = 200 if result.get("status") == "success" else 400
        return jsonify(result), http_code

    @app.route("/api/breast-cancer/predict/clinical", methods=["POST"])
    def predict_clinical_endpoint() -> Response:
        """
        Clinical diagnostic prediction endpoint.
        Accepts complete 30-parameter fine needle aspirate (FNA) measurements.
        """
        data, err_resp = _extract_json_payload(request)
        if err_resp is not None:
            resp, code = err_resp
            return resp, code

        result = bc_service.predict_clinical(data)
        http_code = 200 if result.get("status") == "success" else 400
        return jsonify(result), http_code

    # -------------------------------------------------------------
    # Standardized Error Handlers
    # -------------------------------------------------------------

    @app.errorhandler(400)
    def handle_bad_request(e: Any) -> Response:
        return jsonify({
            "status": "error",
            "disease": DISEASE_KEY,
            "error_type": "invalid_request",
            "message": "Bad request syntax or invalid parameters.",
        }), 400

    @app.errorhandler(404)
    def handle_not_found(e: Any) -> Response:
        return jsonify({
            "status": "error",
            "disease": DISEASE_KEY,
            "error_type": "not_found",
            "message": "Endpoint not found.",
        }), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(e: Any) -> Response:
        return jsonify({
            "status": "error",
            "disease": DISEASE_KEY,
            "error_type": "method_not_allowed",
            "message": "Method not allowed for requested URL.",
        }), 405

    @app.errorhandler(500)
    def handle_internal_error(e: Any) -> Response:
        return jsonify({
            "status": "error",
            "disease": DISEASE_KEY,
            "error_type": "internal_error",
            "message": "An internal server error occurred.",
        }), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)
