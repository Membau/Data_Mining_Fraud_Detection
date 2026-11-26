import json
import os
import pickle
import tempfile
import numpy as np
from django.test import Client, TestCase
from django.urls import reverse
from sklearn.preprocessing import StandardScaler
import xgboost as xgb_lib
from fraud_app.ml_service import FraudPredictionService


class FraudAppMLTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.tmp_dir = tempfile.TemporaryDirectory()
        models_dir = self.tmp_dir.name

        feature_names = FraudPredictionService.get_feature_names()
        rng = np.random.default_rng(0)
        X_dummy = rng.normal(size=(40, len(feature_names)))
        y_dummy = np.array([0] * 35 + [1] * 5)

        scaler = StandardScaler().fit(X_dummy)
        model = xgb_lib.XGBClassifier(n_estimators=5, max_depth=2, random_state=0)
        model.fit(scaler.transform(X_dummy), y_dummy)

        with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(models_dir, "xgb.pkl"), "wb") as f:
            pickle.dump(model, f)

        metrics = {
            "XGBoost": {
                "precision": 0.90,
                "recall": 0.80,
                "f1": 0.85,
                "roc_auc": 0.95,
                "pr_auc": 0.88,
                "confusion_matrix": [[34, 1], [1, 4]],
                "feature_importance": {name: 1.0 / len(feature_names) for name in feature_names},
            }
        }
        with open(os.path.join(models_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f)

        samples = [{"sample_id": 1, "Time": 120.0, "Amount": 49.99, "Class": 0, **{f"V{i}": 0.0 for i in range(1, 29)}}]
        with open(os.path.join(models_dir, "samples.json"), "w", encoding="utf-8") as f:
            json.dump(samples, f)

        self.service = FraudPredictionService(models_dir=models_dir)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_ml_service_prediction_structure(self):
        features = {f"V{i}": 0.1 for i in range(1, 29)}
        features["Time"] = 100.0
        features["Amount"] = 50.0
        result = self.service.predict("XGBoost", features)
        self.assertIn("prediction", result)
        self.assertIn("confidence", result)
        self.assertLess(result["confidence"], 1.0)
        self.assertIn("top_drivers", result)
        self.assertIn("explanation", result)
        for key in ("idea", "guidance", "critique", "evidence"):
            self.assertIn(key, result["explanation"])

    def test_index_view_renders(self):
        response = self.client.get(reverse("index"))
        self.assertEqual(response.status_code, 200)

    def test_predict_api_invalid_json(self):
        response = self.client.post(reverse("predict_api"), data="{bad-json", content_type="application/json")
        self.assertEqual(response.status_code, 400)
