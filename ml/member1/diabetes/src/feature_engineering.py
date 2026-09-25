"""Phase 5 feature-engineering interface for Diabetes.

No target-derived or arbitrary risk-score features are created.
Missing-value handling and learned transformations belong to later training-only preprocessing.
"""

import pandas as pd

TARGET_COLUMN = "Outcome"
EXCLUDED_COLUMNS = {TARGET_COLUMN}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy without changing source data."""
    return df.copy()
