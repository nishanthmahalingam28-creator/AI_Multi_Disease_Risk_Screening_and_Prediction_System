"""
Lung Cancer Prediction Module - Flask REST API Layer.

Exposes the Lung Cancer Prediction Service through standardized REST endpoints:
- GET  /api/lung-cancer/health
- GET  /api/lung-cancer/model-info
- POST /api/lung-cancer/predict/user
- POST /api/lung-cancer/predict/clinical

Delegates all inference, scaling, and validation logic to LungCancerPredictionService.
The API layer acts purely as an HTTP transport adapter and controlled exception boundary.

Medical Safety Disclaimer:
This system provides a machine-learning-based lung cancer risk/screening prediction
from the supplied survey features. It is not a medical diagnosis and must not be used
as a substitute for professional medical evaluation. The model was developed and
evaluated on the supplied dataset and has not been established here as a clinically
validated diagnostic tool.
"""

from typing import Optional, Dict, Any, Tuple
import json
from flask import Flask, request, jsonify, Response

from .service import LungCancerPredictionService, DISEASE_KEY, SERVICE_NAME


class DuplicateKeyError(ValueError):
    """Raised when duplicate JSON keys are detected in request body."""
    pass


def create_app(
    test_config: Optional[Dict[str, Any]] = None,
    service: Optional[LungCancerPredictionService] = None,
) -> Flask:
    """
    Application factory for the Lung Cancer Flask REST API.

    Args:
        test_config: Optional configuration dictionary for testing.
        service: Optional pre-configured LungCancerPredictionService instance.

    Returns:
        Flask: Configured Flask application instance.
    """
    app = Flask(__name__)

    # Default application settings
    app.config["JSON_SORT_KEYS"] = False
    if test_config is not None:
        app.config.update(test_config)

    # Instantiate prediction service if not injected
    lc_service = service if service is not None else LungCancerPredictionService()

    # -------------------------------------------------------------
    # Helper: Safe JSON Body Extraction & Duplicate Key Detection
    # -------------------------------------------------------------
    def _extract_json_payload(req: request) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[Response, int]]]:
        """
        Safely extracts and validates JSON payload from request, detecting
        duplicate keys, malformed JSON, empty bodies, and non-dict JSON types.
        """
        if not req.is_json:
            return None, (jsonify({
                "status": "error",
                "service": SERVICE_NAME,
                "disease": DISEASE_KEY,
                "error_type": "invalid_request",
                "message": "Request body must contain valid JSON with Content-Type application/json.",
            }), 400)

        raw_text = req.get_data(as_text=True)
        if not raw_text or not raw_text.strip():
            return None, (jsonify({
                "status": "error",
                "service": SERVICE_NAME,
                "disease": DISEASE_KEY,
                "error_type": "invalid_request",
                "message": "Request body cannot be empty.",
            }), 400)

        def _detect_duplicates(pairs):
            d = {}
            for k, v in pairs:
                if k in d:
                    raise DuplicateKeyError(f"Duplicate JSON field: {k}")
                d[k] = v
            return d

        try:
            parsed = json.loads(raw_text, object_pairs_hook=_detect_duplicates)
        except DuplicateKeyError as de:
            return None, (jsonify({
                "status": "error",
                "service": SERVICE_NAME,
                "disease": DISEASE_KEY,
                "error_type": "validation_error",
                "message": str(de),
            }), 400)
        except json.JSONDecodeError:
            return None, (jsonify({
                "status": "error",
                "service": SERVICE_NAME,
                "disease": DISEASE_KEY,
                "error_type": "invalid_request",
                "message": "Malformed JSON in request body.",
            }), 400)
        except Exception:
            return None, (jsonify({
                "status": "error",
                "service": SERVICE_NAME,
                "disease": DISEASE_KEY,
                "error_type": "invalid_request",
                "message": "Invalid JSON format in request body.",
            }), 400)

        if not isinstance(parsed, dict):
            return None, (jsonify({
                "status": "error",
                "service": SERVICE_NAME,
                "disease": DISEASE_KEY,
                "error_type": "invalid_request",
                "message": "Request body must contain a valid JSON object.",
            }), 400)

        return parsed, None

    # -------------------------------------------------------------
    # REST API Routes
    # -------------------------------------------------------------

    @app.route("/api/lung-cancer/health", methods=["GET"])
    def health_check() -> Tuple[Response, int]:
        """
        Health and capability check endpoint.
        Returns 200 OK if model and preprocessor are ready, 503 if unavailable.
        """
        health_status = lc_service.health_check()
        http_code = 200 if health_status.get("status") == "ready" else 503
        return jsonify(health_status), http_code

    @app.route("/api/lung-cancer/model-info", methods=["GET"])
    def model_info() -> Tuple[Response, int]:
        """
        Model metadata endpoint.
        Returns active model identity, threshold, feature count, and class mapping.
        """
        info = lc_service.get_model_info()
        http_code = 200 if info.get("status") == "success" else 503
        return jsonify(info), http_code

    @app.route("/api/lung-cancer/predict/user", methods=["POST"])
    def predict_user_endpoint() -> Tuple[Response, int]:
        """
        User-facing screening prediction endpoint.
        Accepts 15 survey features and delegates to LungCancerPredictionService.predict_user.
        """
        data, err_resp = _extract_json_payload(request)
        if err_resp is not None:
            return err_resp

        try:
            result = lc_service.predict_user(data)
            if result.get("status") == "success":
                return jsonify(result), 200
            elif result.get("error_type") in ("service_error", "artifact_error", "transformation_error", "inference_error"):
                return jsonify(result), 503
            else:
                return jsonify(result), 400
        except Exception:
            return jsonify({
                "status": "error",
                "service": SERVICE_NAME,
                "disease": DISEASE_KEY,
                "error_type": "internal_error",
                "message": "An internal server error occurred.",
            }), 500

    @app.route("/api/lung-cancer/predict/clinical", methods=["POST"])
    def predict_clinical_endpoint() -> Tuple[Response, int]:
        """
        Clinical screening prediction endpoint.
        Accepts 15 clinical survey features and delegates to LungCancerPredictionService.predict_clinical.
        """
        data, err_resp = _extract_json_payload(request)
        if err_resp is not None:
            return err_resp

        try:
            result = lc_service.predict_clinical(data)
            if result.get("status") == "success":
                return jsonify(result), 200
            elif result.get("error_type") in ("service_error", "artifact_error", "transformation_error", "inference_error"):
                return jsonify(result), 503
            else:
                return jsonify(result), 400
        except Exception:
            return jsonify({
                "status": "error",
                "service": SERVICE_NAME,
                "disease": DISEASE_KEY,
                "error_type": "internal_error",
                "message": "An internal server error occurred.",
            }), 500

    # -------------------------------------------------------------
    # Standardized Error Handlers
    # -------------------------------------------------------------

    @app.errorhandler(400)
    def handle_bad_request(e: Any) -> Tuple[Response, int]:
        return jsonify({
            "status": "error",
            "service": SERVICE_NAME,
            "disease": DISEASE_KEY,
            "error_type": "invalid_request",
            "message": "Bad request syntax or invalid parameters.",
        }), 400

    @app.errorhandler(404)
    def handle_not_found(e: Any) -> Tuple[Response, int]:
        return jsonify({
            "status": "error",
            "service": SERVICE_NAME,
            "disease": DISEASE_KEY,
            "error_type": "not_found",
            "message": "Endpoint not found.",
        }), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(e: Any) -> Tuple[Response, int]:
        return jsonify({
            "status": "error",
            "service": SERVICE_NAME,
            "disease": DISEASE_KEY,
            "error_type": "method_not_allowed",
            "message": "Method not allowed for requested URL.",
        }), 405

    @app.errorhandler(500)
    def handle_internal_error(e: Any) -> Tuple[Response, int]:
        return jsonify({
            "status": "error",
            "service": SERVICE_NAME,
            "disease": DISEASE_KEY,
            "error_type": "internal_error",
            "message": "An internal server error occurred.",
        }), 500

    @app.errorhandler(503)
    def handle_service_unavailable(e: Any) -> Tuple[Response, int]:
        return jsonify({
            "status": "error",
            "service": SERVICE_NAME,
            "disease": DISEASE_KEY,
            "error_type": "service_unavailable",
            "message": "Lung cancer prediction service is currently unavailable.",
        }), 503

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)
