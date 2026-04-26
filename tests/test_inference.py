import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.inference import run_batch_inference


class TestInference(unittest.TestCase):
    def test_batch_inference_writes_output(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            input_csv = tmp / "input.csv"
            model_path = tmp / "model.joblib"
            output_csv = tmp / "out.csv"

            df = pd.DataFrame(
                {
                    "Title": ["A", "B"],
                    "Review Text": ["good", "bad"],
                    "Age": [30, 45],
                    "Positive Feedback Count": [2, 0],
                    "Rating": [5, 2],
                    "Division Name": ["General", "General"],
                    "Department Name": ["Tops", "Bottoms"],
                    "Class Name": ["Blouses", "Pants"],
                }
            )
            df.to_csv(input_csv, index=False)

            # Dummy model for test speed; pipeline interface matches production expectations.
            x = pd.DataFrame({"text": ["a", "b"]})
            y = [1, 0]
            model = Pipeline([("model", DummyClassifier(strategy="most_frequent"))])
            model.fit(x, y)

            import joblib

            joblib.dump(model, model_path)
            out_path = run_batch_inference(input_csv=input_csv, model_path=model_path, output_csv=output_csv)

            self.assertTrue(out_path.exists())
            pred_df = pd.read_csv(out_path)
            self.assertIn("prediction", pred_df.columns)
            self.assertIn("score", pred_df.columns)


if __name__ == "__main__":
    unittest.main()
