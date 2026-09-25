import unittest


class TestMember3Imports(unittest.TestCase):

    def test_imports(self):
        import asthma.train
        import asthma.predict_user
        import asthma.predict_clinical
        import parkinsons.train
        import parkinsons.predict_user
        import parkinsons.predict_clinical


if __name__ == "__main__":
    unittest.main()
