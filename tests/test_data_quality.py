import unittest

import pandas as pd

from src.pipeline.validate import build_data_quality_report, validate_data_gate


class TestDataQuality(unittest.TestCase):
    def test_quality_report_has_expected_keys(self):
        df = pd.DataFrame(
            {
                "Title": ["a", "b"],
                "Review Text": ["good", "bad"],
                "Age": [30, 40],
                "Positive Feedback Count": [1, 0],
                "Rating": [5, 2],
                "Division Name": ["General", "General"],
                "Department Name": ["Tops", "Bottoms"],
                "Class Name": ["Blouses", "Pants"],
                "Recommended IND": [1, 0],
            }
        )
        report = build_data_quality_report(df)
        self.assertIn("row_count", report)
        self.assertIn("target_distribution", report)
        self.assertEqual(report["row_count"], 2)

    def test_gate_fails_for_missing_target(self):
        report = {
            "row_count": 2000,
            "missing_required_columns": ["Recommended IND"],
            "target_distribution": {},
        }
        errors = validate_data_gate(report, min_rows=1000)
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
