import unittest


class TestMember3Imports(unittest.TestCase):

    def test_imports(self):
        import ml.member3.asthma.train
        import ml.member3.asthma.predict_user
        import ml.member3.asthma.predict_clinical
        import ml.member3.parkinsons.train
        import ml.member3.parkinsons.predict_user
        import ml.member3.parkinsons.predict_clinical


if __name__ == "__main__":
    unittest.main()
