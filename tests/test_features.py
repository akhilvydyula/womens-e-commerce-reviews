import unittest

import pandas as pd

from src.data import make_text_feature


class TestFeatures(unittest.TestCase):
    def test_make_text_feature_combines_title_and_review(self):
        df = pd.DataFrame({"Title": ["Nice"], "Review Text": ["Great fit"]})
        text = make_text_feature(df)
        self.assertEqual(text.iloc[0], "Nice Great fit")


if __name__ == "__main__":
    unittest.main()
