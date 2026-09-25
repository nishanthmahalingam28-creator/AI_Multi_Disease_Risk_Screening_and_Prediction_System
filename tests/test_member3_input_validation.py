import json
import unittest
from pathlib import Path

from asthma.predict_user import predict_user as asthma_user
from asthma.predict_clinical import predict_clinical as asthma_clinical
from parkinsons.predict_user import predict_user as parkinsons_user
from parkinsons.predict_clinical import predict_clinical as parkinsons_clinical


ROOT = Path(__file__).resolve().parents[1]


def load_schema(disease):
    return json.loads(
        (ROOT / disease / "user_schema.json").read_text(encoding="utf-8")
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


class TestMember3InputValidation(unittest.TestCase):

    def test_asthma_user_missing_required_feature(self):
        schema = load_schema("asthma")
        features = make_valid_input(schema)
        features.pop("Age")

        with self.assertRaises(Exception):
            asthma_user(features)

    def test_asthma_clinical_missing_required_feature(self):
        schema = load_schema("asthma")
        features = make_valid_input(schema)
        features.pop("Age")

        with self.assertRaises(Exception):
            asthma_clinical(features)

    def test_parkinsons_user_missing_required_feature(self):
        schema = load_schema("parkinsons")
        features = make_valid_input(schema)
        features.pop("MDVP:Fo(Hz)")

        with self.assertRaises(Exception):
            parkinsons_user(features)

    def test_parkinsons_clinical_missing_required_feature(self):
        schema = load_schema("parkinsons")
        features = make_valid_input(schema)
        features.pop("MDVP:Fo(Hz)")

        with self.assertRaises(Exception):
            parkinsons_clinical(features)


if __name__ == "__main__":
    unittest.main()
