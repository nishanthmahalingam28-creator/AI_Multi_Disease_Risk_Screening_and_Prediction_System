"""
Member 3B Unified Cancer Prediction Module - Flask REST API Layer.

Exposes unified cancer prediction endpoints:
- GET  /api/member3b/health
- GET  /api/member3b/model-info
- POST /api/member3b/predict

Delegates all disease-specific validation and inference to CancerPredictionService.
Acts purely as a transport adapter and controlled exception boundary.

Medical Safety Disclaimer:
This system provides machine-learning-based cancer risk screening predictions
and risk estimates. It is not a medical diagnostic tool and must not replace
professional medical evaluation by qualified healthcare providers.
"""

from typing import Optional, Dict, Any, Tuple
import json
from flask import Flask, request, jsonify, Response

from .service import (
    CancerPredictionService,
    UNIFIED_SERVICE_NAME,
    SUPPORTED_DISEASES,
)


class DuplicateKeyError(ValueError):
    """Raised when duplicate JSON keys are detected in request body."""
    pass


def create_app(
    test_config: Optional[Dict[str, Any]] = None,
    service: Optional[CancerPredictionService] = None,
) -> Flask:
    """
    Application factory for the Member 3B Unified Cancer REST API.

    Args:
        test_config: Optional configuration dictionary for testing.
        service: Optional pre-configured CancerPredictionService instance.

    Returns:
        Flask: Configured Flask application instance.
    """
    app = Flask(__name__)

    # Default settings
    app.config["JSON_SORT_KEYS"] = False
    if test_config is not None:
        app.config.update(test_config)

    # Instantiate unified service if not injected
    unified_service = service if service is not None else CancerPredictionService()

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
                "service": UNIFIED_SERVICE_NAME,
                "error_type": "invalid_request",
                "message": "Request body must contain valid JSON with Content-Type application/json.",
            }), 400)

        raw_text = req.get_data(as_text=True)
        if not raw_text or not raw_text.strip():
            return None, (jsonify({
                "status": "error",
                "service": UNIFIED_SERVICE_NAME,
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
                "service": UNIFIED_SERVICE_NAME,
                "error_type": "validation_error",
                "message": str(de),
            }), 400)
        except json.JSONDecodeError:
            return None, (jsonify({
                "status": "error",
                "service": UNIFIED_SERVICE_NAME,
                "error_type": "invalid_request",
                "message": "Malformed JSON in request body.",
            }), 400)
        except Exception:
            return None, (jsonify({
                "status": "error",
                "service": UNIFIED_SERVICE_NAME,
                "error_type": "invalid_request",
                "message": "Invalid JSON format in request body.",
            }), 400)

        if not isinstance(parsed, dict):
            return None, (jsonify({
                "status": "error",
                "service": UNIFIED_SERVICE_NAME,
                "error_type": "invalid_request",
                "message": "Request body must contain a valid JSON object.",
            }), 400)

        return parsed, None

    # -------------------------------------------------------------
    # REST API Routes
    # -------------------------------------------------------------

    @app.route("/api/member3b/health", methods=["GET"])
    def unified_health_check() -> Tuple[Response, int]:
        """
        Unified capability and readiness check endpoint.
        Returns 200 OK if all underlying disease models are ready, 503 if unavailable.
        """
        health_status = unified_service.health_check()
        http_code = 200 if health_status.get("status") == "ready" else 503
        return jsonify(health_status), http_code

    @app.route("/api/member3b/model-info", methods=["GET"])
    def unified_model_info() -> Tuple[Response, int]:
        """
        Unified model metadata endpoint.
        Safely aggregates model details across all supported cancer prediction services.
        """
        info = unified_service.get_model_info()
        http_code = 200 if info.get("status") == "success" else 503
        return jsonify(info), http_code

    @app.route("/api/member3b/predict", methods=["POST"])
    def unified_predict_endpoint() -> Tuple[Response, int]:
        """
        Unified cancer risk screening prediction endpoint.
        Accepts envelope with 'disease' and 'data', routing to the appropriate service.
        """
        payload, err_resp = _extract_json_payload(request)
        if err_resp is not None:
            return err_resp

        # Validate envelope presence
        if "disease" not in payload:
            return jsonify({
                "status": "error",
                "service": UNIFIED_SERVICE_NAME,
                "error_type": "validation_error",
                "message": "Missing required field 'disease'.",
            }), 400

        if "data" not in payload:
            return jsonify({
                "status": "error",
                "service": UNIFIED_SERVICE_NAME,
                "error_type": "validation_error",
                "message": "Missing required field 'data'.",
            }), 400

        disease = payload.get("disease")
        data = payload.get("data")
        mode = payload.get("mode", "user")

        try:
            result = unified_service.predict(disease=disease, data=data, mode=mode)
            if result.get("status") == "success":
                return jsonify(result), 200
            elif result.get("error_type") in ("service_error", "service_unavailable", "artifact_error"):
                return jsonify(result), 503
            else:
                return jsonify(result), 400
        except Exception:
            return jsonify({
                "status": "error",
                "service": UNIFIED_SERVICE_NAME,
                "error_type": "internal_error",
                "message": "An internal server error occurred.",
            }), 500

    # -------------------------------------------------------------
    # Disease-Specific REST API Routes (Direct Access Pass-Through)
    # -------------------------------------------------------------

    @app.route("/api/breast-cancer/health", methods=["GET"])
    def breast_health_check() -> Tuple[Response, int]:
        health_status = unified_service.breast_service.health_check()
        http_code = 200 if health_status.get("status") == "ready" else 503
        return jsonify(health_status), http_code

    @app.route("/api/breast-cancer/model-info", methods=["GET"])
    def breast_model_info() -> Tuple[Response, int]:
        info = unified_service.breast_service.get_model_info()
        return jsonify(info), 200

    @app.route("/api/breast-cancer/predict/user", methods=["POST"])
    def breast_predict_user() -> Tuple[Response, int]:
        data, err_resp = _extract_json_payload(request)
        if err_resp is not None:
            return err_resp
        result = unified_service.breast_service.predict_user(data)
        http_code = 200 if result.get("status") == "success" else 400
        return jsonify(result), http_code

    @app.route("/api/breast-cancer/predict/clinical", methods=["POST"])
    def breast_predict_clinical() -> Tuple[Response, int]:
        data, err_resp = _extract_json_payload(request)
        if err_resp is not None:
            return err_resp
        result = unified_service.breast_service.predict_clinical(data)
        http_code = 200 if result.get("status") == "success" else 400
        return jsonify(result), http_code

    @app.route("/api/lung-cancer/health", methods=["GET"])
    def lung_health_check() -> Tuple[Response, int]:
        health_status = unified_service.lung_service.health_check()
        http_code = 200 if health_status.get("status") == "ready" else 503
        return jsonify(health_status), http_code

    @app.route("/api/lung-cancer/model-info", methods=["GET"])
    def lung_model_info() -> Tuple[Response, int]:
        info = unified_service.lung_service.get_model_info()
        http_code = 200 if info.get("status") == "success" else 503
        return jsonify(info), http_code

    @app.route("/api/lung-cancer/predict/user", methods=["POST"])
    def lung_predict_user() -> Tuple[Response, int]:
        data, err_resp = _extract_json_payload(request)
        if err_resp is not None:
            return err_resp
        result = unified_service.lung_service.predict_user(data)
        if result.get("status") == "success":
            return jsonify(result), 200
        elif result.get("error_type") in ("service_error", "artifact_error", "transformation_error", "inference_error"):
            return jsonify(result), 503
        else:
            return jsonify(result), 400

    @app.route("/api/lung-cancer/predict/clinical", methods=["POST"])
    def lung_predict_clinical() -> Tuple[Response, int]:
        data, err_resp = _extract_json_payload(request)
        if err_resp is not None:
            return err_resp
        result = unified_service.lung_service.predict_clinical(data)
        if result.get("status") == "success":
            return jsonify(result), 200
        elif result.get("error_type") in ("service_error", "artifact_error", "transformation_error", "inference_error"):
            return jsonify(result), 503
        else:
            return jsonify(result), 400

    # -------------------------------------------------------------
    # Standardized Error Handlers
    # -------------------------------------------------------------

    @app.errorhandler(400)
    def handle_bad_request(e: Any) -> Tuple[Response, int]:
        return jsonify({
            "status": "error",
            "service": UNIFIED_SERVICE_NAME,
            "error_type": "invalid_request",
            "message": "Bad request syntax or invalid parameters.",
        }), 400

    @app.errorhandler(404)
    def handle_not_found(e: Any) -> Tuple[Response, int]:
        return jsonify({
            "status": "error",
            "service": UNIFIED_SERVICE_NAME,
            "error_type": "not_found",
            "message": "Endpoint not found.",
        }), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(e: Any) -> Tuple[Response, int]:
        return jsonify({
            "status": "error",
            "service": UNIFIED_SERVICE_NAME,
            "error_type": "method_not_allowed",
            "message": "Method not allowed for requested URL.",
        }), 405

    @app.errorhandler(500)
    def handle_internal_error(e: Any) -> Tuple[Response, int]:
        return jsonify({
            "status": "error",
            "service": UNIFIED_SERVICE_NAME,
            "error_type": "internal_error",
            "message": "An internal server error occurred.",
        }), 500

    @app.errorhandler(503)
    def handle_service_unavailable(e: Any) -> Tuple[Response, int]:
        return jsonify({
            "status": "error",
            "service": UNIFIED_SERVICE_NAME,
            "error_type": "service_unavailable",
            "message": "Unified cancer prediction service is currently unavailable.",
        }), 503

    return app


if __name__ == "__main__":
    from .config import get_config
    cfg = get_config()
    app = create_app()
    app.run(host=cfg.HOST, port=cfg.PORT, debug=cfg.DEBUG)
