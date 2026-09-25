"""Phase 5 feature-engineering interface for Stroke.

The identifier id is excluded from predictive features by Phase 3.
Phase 5 does not create target-derived features or learned transforms.
"""

import pandas as pd

TARGET_COLUMN = "stroke"
EXCLUDED_COLUMNS = {TARGET_COLUMN, "id"}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy and exclude id if it is accidentally supplied."""
    result = df.copy()
    if "id" in result.columns:
        result = result.drop(columns=["id"])
    return result
