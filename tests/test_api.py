import os
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.api import get_app
from src.inference import prepare_features_for_model


class TestApi(unittest.TestCase):
    def test_health_and_predict(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "model.joblib"
            row = {
                "Title": "Nice",
                "Review Text": "Fits well",
                "Age": 30,
                "Positive Feedback Count": 1,
                "Rating": 5,
                "Division Name": "General",
                "Department Name": "Tops",
                "Class Name": "Knits",
            }
            feat = prepare_features_for_model(pd.DataFrame([row]))
            model = Pipeline([("model", DummyClassifier(strategy="constant", constant=1))])
            model.fit(feat, [1])
            joblib.dump(model, model_path)

            os.environ["MODEL_PATH"] = str(model_path)
            with TestClient(get_app()) as client:
                root = client.get("/")
                self.assertEqual(root.status_code, 200)
                self.assertIn("interactive_docs", root.json())
                self.assertEqual(root.json().get("browser_playground"), "/ui")

                ui = client.get("/ui")
                self.assertEqual(ui.status_code, 200)
                self.assertIn("text/html", ui.headers.get("content-type", ""))
                self.assertIn("Review inference playground", ui.text)
                self.assertIn('"/predict"', ui.text)

                h = client.get("/health")
                self.assertEqual(h.status_code, 200)
                self.assertEqual(h.json().get("status"), "ok")

                p = client.post("/predict", json=row)
                self.assertEqual(p.status_code, 200)
                body = p.json()
                self.assertIn("prediction", body)
                self.assertIn("score", body)


if __name__ == "__main__":
    unittest.main()
