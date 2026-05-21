import unittest

import pandas as pd

from src.config import TARGET_COLUMN
from src.data import stratified_subsample


class TestStratifiedSubsample(unittest.TestCase):
    def test_frac_reduces_rows_preserves_binary_target(self):
        df = pd.DataFrame(
            {
                "Title": [f"t{i}" for i in range(200)],
                "Review Text": [f"r{i}" for i in range(200)],
                "Age": [20] * 200,
                "Positive Feedback Count": [0] * 200,
                "Rating": [4] * 200,
                "Division Name": ["G"] * 200,
                "Department Name": ["D"] * 200,
                "Class Name": ["C"] * 200,
                TARGET_COLUMN: [0] * 100 + [1] * 100,
            }
        )
        out = stratified_subsample(df, sample_frac=0.2, random_state=42)
        self.assertEqual(len(out), 40)
        self.assertSetEqual(set(out[TARGET_COLUMN].unique()), {0, 1})

    def test_max_rows(self):
        df = pd.DataFrame({TARGET_COLUMN: [0] * 50 + [1] * 50})
        out = stratified_subsample(df, max_rows=30, random_state=0)
        self.assertEqual(len(out), 30)


if __name__ == "__main__":
    unittest.main()
