import json
import unittest
from pathlib import Path

from ml.member3.asthma.predict_user import predict_user as asthma_user
from ml.member3.asthma.predict_clinical import predict_clinical as asthma_clinical
from ml.member3.parkinsons.predict_user import predict_user as parkinsons_user
from ml.member3.parkinsons.predict_clinical import predict_clinical as parkinsons_clinical


ROOT = Path(__file__).resolve().parents[1]


def load_schema(disease, module):
    return json.loads(
        (ROOT / disease / f"{module}_schema.json").read_text(encoding="utf-8")
    )


def make_valid_input(schema):
    values = {}

    for feature in schema["features"]:
        name = feature["name"]
        dtype = feature["dtype"]

        if dtype == "str":
            values[name] = "Unknown"
        elif dtype == "int64":
            values[name] = 1
        elif dtype == "float64":
            values[name] = 1.0
        else:
            raise ValueError(f"Unsupported dtype in schema: {dtype}")

    return values


class TestMember3Predictions(unittest.TestCase):

    def test_asthma_user_prediction(self):
        schema = load_schema("asthma", "user")
        result = asthma_user(make_valid_input(schema))

        self.assertEqual(result["disease"], "asthma")
        self.assertEqual(result["module"], "user")
        self.assertIn(result["prediction"], (0, 1))
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)
        self.assertIn("risk_category", result)
        self.assertIn("disclaimer", result)

    def test_asthma_clinical_prediction(self):
        schema = load_schema("asthma", "clinical")
        result = asthma_clinical(make_valid_input(schema))

        self.assertEqual(result["disease"], "asthma")
        self.assertEqual(result["module"], "clinical")
        self.assertIn(result["prediction"], (0, 1))
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)

    def test_parkinsons_user_prediction(self):
        schema = load_schema("parkinsons", "user")
        result = parkinsons_user(make_valid_input(schema))

        self.assertEqual(result["disease"], "parkinsons")
        self.assertEqual(result["module"], "user")
        self.assertIn(result["prediction"], (0, 1))
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)
        self.assertIn("risk_category", result)
        self.assertIn("disclaimer", result)

    def test_parkinsons_clinical_prediction(self):
        schema = load_schema("parkinsons", "clinical")
        result = parkinsons_clinical(make_valid_input(schema))

        self.assertEqual(result["disease"], "parkinsons")
        self.assertEqual(result["module"], "clinical")
        self.assertIn(result["prediction"], (0, 1))
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
