"""Phase 5 feature-engineering interface for Heart Disease.

No learned transformation or new predictor is created in Phase 5.
Later preprocessing should fit encoders/scalers only on training data.
"""

import pandas as pd

TARGET_COLUMN = "Heart Disease"
EXCLUDED_COLUMNS = {TARGET_COLUMN}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy without target-derived or arbitrary engineered features."""
    return df.copy()
