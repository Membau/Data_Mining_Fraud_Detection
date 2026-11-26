import json
import os
import pickle
import time
import numpy as np
import pandas as pd
import xgboost as xgb_lib
from river import forest, tree
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def load_prefix_suffix(prefix_path: str, suffix_path: str):
    """Load prefix (initial training) and suffix (streaming adaptation & evaluation)."""
    prefix = pd.read_csv(prefix_path)
    suffix = pd.read_csv(suffix_path)
    common_cols = [c for c in prefix.columns if c in suffix.columns]
    prefix = prefix[common_cols]
    suffix = suffix[common_cols]
    return prefix, suffix


def compute_metrics(y_true, y_pred, y_prob):
    """Compute reliability metrics suitable for imbalanced fraud detection."""
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def compute_river_permutation_importance(
    model, X_eval: np.ndarray, y_eval: np.ndarray, feature_names: list, seed: int = 42
) -> dict:
    """Estimate feature importance for online River models via PR-AUC permutation drop."""
    rng = np.random.default_rng(seed)

    def _eval_pr_auc(X_mat: np.ndarray) -> float:
        probs = []
        for xi in X_mat:
            x_dict = {feature_names[j]: float(xi[j]) for j in range(len(feature_names))}
            p = model.predict_proba_one(x_dict)
            probs.append(float(p.get(1, 0.0)))
        return float(average_precision_score(y_eval, probs))

    baseline_pr = _eval_pr_auc(X_eval)
    importances = {}
    for idx, name in enumerate(feature_names):
        X_perm = X_eval.copy()
        X_perm[:, idx] = rng.permutation(X_perm[:, idx])
        perm_pr = _eval_pr_auc(X_perm)
        importances[name] = max(0.0, float(baseline_pr - perm_pr))

    total = sum(importances.values())
    if total > 0:
        importances = {k: float(v / total) for k, v in importances.items()}
    else:
        importances = {k: 1.0 / len(feature_names) for k in feature_names}
    return importances


def build_14_representative_samples(suffix_df: pd.DataFrame) -> list:
    """Select 14 representative transactions (7 fraud, 7 normal) spanning Amount & Time."""
    frauds = suffix_df[suffix_df["Class"] == 1].sort_values("Amount")
    normals = suffix_df[suffix_df["Class"] == 0].sort_values("Amount")

    fraud_idx = np.linspace(0, len(frauds) - 1, num=min(7, len(frauds)), dtype=int)
    normal_idx = np.linspace(0, len(normals) - 1, num=min(7, len(normals)), dtype=int)

    selected = pd.concat([frauds.iloc[fraud_idx], normals.iloc[normal_idx]], ignore_index=True)
    selected = selected.sample(frac=1.0, random_state=42).reset_index(drop=True)
    records = selected.to_dict(orient="records")
    for idx, rec in enumerate(records, start=1):
        rec["sample_id"] = idx
    return records


def train_and_evaluate(prefix_path: str, suffix_path: str, output_dir: str = "fraud_web/ml_models"):
    os.makedirs(output_dir, exist_ok=True)
    prefix_df, suffix_df = load_prefix_suffix(prefix_path, suffix_path)

    feature_names = [c for c in prefix_df.columns if c != "Class"]
    X_prefix_raw = prefix_df[feature_names].values
    y_prefix = prefix_df["Class"].values.astype(int)

    X_suffix_raw = suffix_df[feature_names].values
    y_suffix = suffix_df["Class"].values.astype(int)

    # Fit StandardScaler strictly on prefix to avoid data leakage
    scaler = StandardScaler()
    X_prefix = scaler.fit_transform(X_prefix_raw)
    X_suffix = scaler.transform(X_suffix_raw)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    metrics = {}

    # 1. XGBoost (Batch Learning on prefix -> evaluate on suffix)
    t0 = time.time()
    neg_count = int(np.sum(y_prefix == 0))
    pos_count = int(np.sum(y_prefix == 1))
    scale_weight = neg_count / max(pos_count, 1)

    xgb_model = xgb_lib.XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    xgb_model.fit(X_prefix, y_prefix)
    xgb_probs = xgb_model.predict_proba(X_suffix)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)

    xgb_imp_raw = xgb_model.feature_importances_
    xgb_imp_sum = float(np.sum(xgb_imp_raw)) or 1.0
    metrics["XGBoost"] = {
        **compute_metrics(y_suffix, xgb_preds, xgb_probs),
        "train_mode": "Batch Learning (train trên prefix, kiểm thử trên suffix)",
        "train_seconds": round(time.time() - t0, 2),
        "feature_importance": {
            feature_names[i]: float(xgb_imp_raw[i] / xgb_imp_sum)
            for i in range(len(feature_names))
        },
    }
    print(f"[XGBoost] Done in {metrics['XGBoost']['train_seconds']}s | PR-AUC={metrics['XGBoost']['pr_auc']:.4f} | F1={metrics['XGBoost']['f1']:.4f}")

    # Stratified subsample of suffix for fast permutation importance of River models
    rng = np.random.default_rng(42)
    fraud_mask = y_suffix == 1
    normal_indices = np.where(~fraud_mask)[0]
    sampled_normal_idx = rng.choice(normal_indices, size=min(1500, len(normal_indices)), replace=False)
    eval_sub_idx = np.concatenate([np.where(fraud_mask)[0], sampled_normal_idx])
    X_perm_sub = X_suffix[eval_sub_idx]
    y_perm_sub = y_suffix[eval_sub_idx]

    # 2. HAT: Hoeffding Adaptive Tree (Pre-train prefix -> Prequential test-then-train suffix)
    # Apply online minority-class weighting (repeat=15 for fraud) to prevent majority-class collapse
    t0 = time.time()
    hat = tree.HoeffdingAdaptiveTreeClassifier(
        grace_period=50,
        delta=0.01,
        leaf_prediction="nb",
        seed=42,
    )
    for xi, yi in zip(X_prefix, y_prefix):
        x_dict = {feature_names[j]: float(xi[j]) for j in range(len(feature_names))}
        for _ in range(15 if yi == 1 else 1):
            hat.learn_one(x_dict, int(yi))

    hat_preds, hat_probs = [], []
    for xi, yi in zip(X_suffix, y_suffix):
        x_dict = {feature_names[j]: float(xi[j]) for j in range(len(feature_names))}
        pred = hat.predict_one(x_dict)
        probs = hat.predict_proba_one(x_dict)
        hat_preds.append(int(pred) if pred is not None else 0)
        hat_probs.append(float(probs.get(1, 0.0)))
        for _ in range(15 if yi == 1 else 1):
            hat.learn_one(x_dict, int(yi))

    metrics["HAT"] = {
        **compute_metrics(y_suffix, np.array(hat_preds), np.array(hat_probs)),
        "train_mode": "Online Streaming Prequential + Minority Weighting (prefix -> suffix)",
        "train_seconds": round(time.time() - t0, 2),
        "feature_importance": compute_river_permutation_importance(hat, X_perm_sub, y_perm_sub, feature_names),
    }
    print(f"[HAT] Done in {metrics['HAT']['train_seconds']}s | PR-AUC={metrics['HAT']['pr_auc']:.4f} | F1={metrics['HAT']['f1']:.4f}")

    # 3. ARF: Adaptive Random Forest (Pre-train prefix -> Prequential test-then-train suffix)
    t0 = time.time()
    arf = forest.ARFClassifier(
        n_models=5,
        grace_period=200,
        seed=42,
    )
    for xi, yi in zip(X_prefix, y_prefix):
        arf.learn_one({feature_names[j]: float(xi[j]) for j in range(len(feature_names))}, int(yi))

    arf_preds, arf_probs = [], []
    for xi, yi in zip(X_suffix, y_suffix):
        x_dict = {feature_names[j]: float(xi[j]) for j in range(len(feature_names))}
        pred = arf.predict_one(x_dict)
        probs = arf.predict_proba_one(x_dict)
        arf_preds.append(int(pred) if pred is not None else 0)
        arf_probs.append(float(probs.get(1, 0.0)))
        arf.learn_one(x_dict, int(yi))

    metrics["ARF"] = {
        **compute_metrics(y_suffix, np.array(arf_preds), np.array(arf_probs)),
        "train_mode": "Online Ensemble Streaming (khởi tạo prefix, thích ứng ADWIN trên suffix)",
        "train_seconds": round(time.time() - t0, 2),
        "feature_importance": compute_river_permutation_importance(arf, X_perm_sub, y_perm_sub, feature_names),
    }
    print(f"[ARF] Done in {metrics['ARF']['train_seconds']}s | PR-AUC={metrics['ARF']['pr_auc']:.4f} | F1={metrics['ARF']['f1']:.4f}")

    # Save dataset summary for UI
    dataset_summary = {
        "prefix_rows": int(len(prefix_df)),
        "prefix_fraud": int(y_prefix.sum()),
        "suffix_rows": int(len(suffix_df)),
        "suffix_fraud": int(y_suffix.sum()),
        "total_features": len(feature_names),
    }
    with open(os.path.join(output_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "xgb.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)
    with open(os.path.join(output_dir, "hat.pkl"), "wb") as f:
        pickle.dump(hat, f)
    with open(os.path.join(output_dir, "arf.pkl"), "wb") as f:
        pickle.dump(arf, f)

    samples = build_14_representative_samples(suffix_df)
    with open(os.path.join(output_dir, "samples.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print("All models, metrics, and 14 representative samples saved to", output_dir)
    return metrics


if __name__ == "__main__":
    train_and_evaluate(
        prefix_path=os.path.join("data", "creditcard_prefix.csv"),
        suffix_path=os.path.join("data", "creditcard_suffix.csv"),
        output_dir=os.path.join("fraud_web", "ml_models"),
    )
