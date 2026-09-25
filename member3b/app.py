"""
Member 3B Unified Cancer Prediction Backend - Application Entry Point.

Provides a unified server exposing:
- GET  /api/member3b/health
- GET  /api/member3b/model-info
- POST /api/member3b/predict
- GET  /api/breast-cancer/health
- GET  /api/breast-cancer/model-info
- POST /api/breast-cancer/predict/user
- POST /api/breast-cancer/predict/clinical
- GET  /api/lung-cancer/health
- GET  /api/lung-cancer/model-info
- POST /api/lung-cancer/predict/user
- POST /api/lung-cancer/predict/clinical

Execution:
    python -m member3b.app
    or
    python member3b/app.py

Medical Safety Disclaimer:
This system provides machine-learning-based cancer risk screening predictions
and risk estimates. It is not a medical diagnostic tool and must not replace
professional medical evaluation by qualified healthcare providers.
"""

import os
import sys
from pathlib import Path

# Ensure repository root is in sys.path for direct execution from any directory
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from member3b.common.config import get_config, validate_port
from member3b.common.api import create_app

# Instantiate application factory
config = get_config()
app = create_app()

if __name__ == "__main__":
    env_port_raw = os.getenv("PORT")
    if env_port_raw is not None:
        try:
            startup_port = validate_port(env_port_raw)
        except ValueError as err:
            sys.exit(f"Configuration error: {err}")
    else:
        startup_port = config.PORT

    print(
        f"Starting Member 3B Unified Cancer Prediction Backend on "
        f"{config.HOST}:{startup_port} (debug={config.DEBUG}, env={config.ENV})..."
    )
    app.run(
        host=config.HOST,
        port=startup_port,
        debug=config.DEBUG,
    )

