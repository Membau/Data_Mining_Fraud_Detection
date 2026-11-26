import json
import os
import pickle
import numpy as np


_MODEL_DESCRIPTIONS = {
    "XGBoost": {
        "idea": "XGBoost xây dựng chuỗi cây quyết định tăng cường độ dốc (Gradient Boosting) theo chế độ Batch trên toàn bộ tập creditcard_prefix.csv, kết hợp tham số scale_pos_weight để phạt nặng sai số trên lớp gian lận thiểu số (~0.19%).",
        "critique": "XGBoost đạt độ chính xác tách lớp rất cao trên dữ liệu tĩnh, nhưng không tự cập nhật trọng số khi phân phối giao dịch thay đổi theo thời gian (Concept Drift) nếu không huấn luyện lại (retrain).",
    },
    "HAT": {
        "idea": "Hoeffding Adaptive Tree (HAT) xử lý từng giao dịch theo dòng (Streaming) dựa trên bất đẳng thức Hoeffding để chọn ngưỡng chia nhánh và tích hợp bộ phát hiện trôi khái niệm ADWIN để tự thay thế nhánh cây lỗi thời.",
        "critique": "Một cây đơn lẻ HAT có tốc độ thích ứng rất nhanh và tiêu tốn ít bộ nhớ, nhưng dễ nhạy cảm với nhiễu cục bộ hoặc tỷ lệ mất cân bằng cực đoan hơn so với mô hình quần thể (Ensemble).",
    },
    "ARF": {
        "idea": "Adaptive Random Forest (ARF) kết hợp nhiều cây Hoeffding song song trên các tập con đặc trưng ngẫu nhiên, đồng thời gắn bộ giám sát ADWIN cho từng cây để tái huấn luyện cây nền (background tree) khi phát hiện Concept Drift.",
        "critique": "ARF cân bằng tốt giữa độ ổn định của Random Forest và khả năng thích ứng luồng dữ liệu, nhưng chi phí tính toán khi huấn luyện tuần tự cao hơn cây đơn HAT.",
    },
}


class FraudPredictionService:
    _instance = None

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), "..", "ml_models")
        self.models_dir = os.path.abspath(models_dir)
        self.models = {
            "XGBoost": self._load_pickle("xgb.pkl"),
            "HAT": self._load_pickle("hat.pkl"),
            "ARF": self._load_pickle("arf.pkl"),
        }
        self.scaler = self._load_pickle("scaler.pkl")
        self.metrics = self._load_json("metrics.json", {})
        self.samples = self._load_json("samples.json", [])
        self.dataset_summary = self._load_json("dataset_summary.json", {})

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_pickle(self, filename: str):
        path = os.path.join(self.models_dir, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _load_json(self, filename: str, default):
        path = os.path.join(self.models_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return default

    @staticmethod
    def get_feature_names():
        return ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    def _top_transaction_drivers(self, model_name: str, feature_names: list, raw_vec: np.ndarray, scaled_vec: np.ndarray):
        importances = self.metrics.get(model_name, {}).get("feature_importance", {})
        drivers = []
        for idx, name in enumerate(feature_names):
            imp = float(importances.get(name, 1.0 / len(feature_names)))
            z_val = float(scaled_vec[idx])
            score = abs(z_val) * (imp + 1e-4)
            drivers.append(
                {
                    "feature": name,
                    "raw_value": round(float(raw_vec[idx]), 4),
                    "z_score": round(z_val, 3),
                    "importance": round(imp, 4),
                    "impact_score": round(score, 4),
                }
            )
        drivers.sort(key=lambda d: d["impact_score"], reverse=True)
        return drivers[:5]

    def predict(self, model_name: str, features_dict: dict) -> dict:
        model = self.models.get(model_name)
        if model is None:
            return {"error": f"Mô hình '{model_name}' chưa được nạp hoặc chưa được huấn luyện."}

        feature_names = self.get_feature_names()
        raw_vec = np.array([float(features_dict.get(feat, 0.0)) for feat in feature_names], dtype=float)
        scaled_mat = self.scaler.transform(raw_vec.reshape(1, -1)) if self.scaler is not None else raw_vec.reshape(1, -1)
        scaled_vec = scaled_mat[0]

        if model_name == "XGBoost":
            probability = float(model.predict_proba(scaled_mat)[0][1])
            prediction = 1 if probability >= 0.5 else 0
        else:
            x_dict = {feature_names[j]: float(scaled_vec[j]) for j in range(len(feature_names))}
            pred = model.predict_one(x_dict)
            probs = model.predict_proba_one(x_dict)
            probability = float(probs.get(1, 0.0))
            prediction = int(pred) if pred is not None else (1 if probability >= 0.5 else 0)

        confidence = float(max(probability, 1.0 - probability))
        label = "Gian lận (Fraud)" if prediction == 1 else "Hợp lệ (Normal)"
        top_drivers = self._top_transaction_drivers(model_name, feature_names, raw_vec, scaled_vec)
        driver_text = ", ".join(
            f"{d['feature']}={d['raw_value']} (z={d['z_score']:+.2f}, imp={d['importance']*100:.1f}%)"
            for d in top_drivers[:3]
        )

        desc = _MODEL_DESCRIPTIONS.get(model_name, _MODEL_DESCRIPTIONS["XGBoost"])
        model_metrics = self.metrics.get(model_name, {})
        pr_auc = model_metrics.get("pr_auc", 0.0)
        f1 = model_metrics.get("f1", 0.0)

        explanation = {
            "idea": desc["idea"],
            "guidance": (
                f"Giao dịch (Amount=${raw_vec[-1]:.2f}, Time={raw_vec[0]:.0f}s) được chuẩn hóa bằng StandardScaler "
                f"(khớp phân phối tập prefix) và đưa qua mô hình {model_name}. Các đặc trưng biến động mạnh nhất: {driver_text}."
            ),
            "critique": (
                f"{desc['critique']} Với xác suất gian lận P(Fraud)={probability*100:.2f}%, "
                f"cần đối chiếu thêm ngưỡng PR-AUC ({pr_auc:.3f}) và F1 ({f1*100:.1f}%) khi ra quyết định khóa thẻ tự động."
            ),
            "evidence": (
                f"Mô hình {model_name} kết luận '{label}' với độ tin cậy {confidence*100:.2f}% "
                f"(P(Fraud)={probability*100:.2f}%), dựa trên đóng góp lớn nhất từ {driver_text}."
            ),
        }

        return {
            "model": model_name,
            "prediction": prediction,
            "label": label,
            "probability": probability,
            "confidence": confidence,
            "top_drivers": top_drivers,
            "explanation": explanation,
        }
