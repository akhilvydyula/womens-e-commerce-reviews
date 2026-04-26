import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pipeline.etl import run_etl


class TestEtl(unittest.TestCase):
    def test_etl_writes_clean_csv_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw.csv"
            out = Path(td) / "clean.csv"
            df = pd.DataFrame(
                {
                    "Title": ["a", "b", "c"],
                    "Review Text": ["good", "bad", "ok"],
                    "Age": [30, 40, 25],
                    "Positive Feedback Count": [1, 0, 2],
                    "Rating": [5, 2, 4],
                    "Division Name": ["General", "General", "General"],
                    "Department Name": ["Tops", "Bottoms", "Tops"],
                    "Class Name": ["Blouses", "Pants", "Knits"],
                    "Recommended IND": [1, 0, 1],
                }
            )
            df.to_csv(raw, index=False)

            clean_path, manifest_path = run_etl(
                raw_path=raw,
                output_path=out,
                min_rows=3,
                fail_on_gate=True,
                include_text_column=True,
            )

            self.assertTrue(clean_path.exists())
            self.assertTrue(manifest_path.exists())
            loaded = pd.read_csv(clean_path)
            self.assertIn("text", loaded.columns)
            self.assertEqual(len(loaded), 3)


if __name__ == "__main__":
    unittest.main()
