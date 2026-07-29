"""
SpoilGuard — Feature Engineering + Model Training
==================================================

Trains two models on the shipment dataset:
  1. Classification -> spoilage_risk (Low / Medium / High)
  2. Regression      -> remaining_shelf_life_hr

Compares Random Forest vs XGBoost for each, does light hyperparameter
tuning (RandomizedSearchCV), evaluates, generates SHAP explainability
for the best classifier, and saves all artifacts to models/saved_models/.

Run:
    python train_models.py --data ../data/raw/shipments.csv
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(df, train_idx=None, route_risk_map=None):
    """Adds engineered features. route_risk_map is fit on TRAIN data only
    (to avoid target leakage) and reused for the test split."""
    df = df.copy()
    df["route"] = df["origin"] + "_" + df["destination"]
    df["transit_vs_shelf_life_ratio"] = (
        df["transit_duration_hr"] / df["base_shelf_life_hr"]
    )
    df["humidity_deviation"] = (df["humidity_pct"] - 65).clip(lower=0)
    df["temp_out_of_band"] = (
        (df["effective_transit_temp_c"] < df["safe_temp_min_c"])
        | (df["effective_transit_temp_c"] > df["safe_temp_max_c"])
    ).astype(int)

    # Route risk score: historical avg spoilage rate, fit on train only
    if route_risk_map is None:
        train_df = df.loc[train_idx] if train_idx is not None else df
        risk_numeric = train_df["spoilage_risk"].map({"Low": 0, "Medium": 0.5, "High": 1})
        route_risk_map = train_df.assign(risk_numeric=risk_numeric).groupby("route")[
            "risk_numeric"
        ].mean().to_dict()
        global_mean = risk_numeric.mean()
    else:
        global_mean = np.mean(list(route_risk_map.values()))

    df["route_risk_score"] = df["route"].map(route_risk_map).fillna(global_mean)
    return df, route_risk_map


CATEGORICAL_COLS = ["product_category", "vehicle_type", "packaging_quality"]
NUMERIC_FEATURES = [
    "distance_km", "transit_duration_hr", "ambient_temp_c", "humidity_pct",
    "effective_transit_temp_c", "temp_excursion_hours", "base_shelf_life_hr",
    "transit_vs_shelf_life_ratio", "humidity_deviation", "temp_out_of_band",
    "route_risk_score",
]


def build_feature_matrix(df):
    X = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=False)
    X[NUMERIC_FEATURES] = df[NUMERIC_FEATURES]
    return X


# ---------------------------------------------------------------------------
# Classification: spoilage_risk
# ---------------------------------------------------------------------------
def train_classification(X_train, X_test, y_train, y_test, feature_names):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    results = {}

    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    rf_params = {
        "n_estimators": [200, 400, 600],
        "max_depth": [6, 10, 15, None],
        "min_samples_leaf": [1, 2, 4],
    }
    rf_search = RandomizedSearchCV(
        rf, rf_params, n_iter=15, cv=3, scoring="f1_macro",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_search.fit(X_train, y_train_enc)
    results["RandomForest"] = rf_search.best_estimator_

    xgb = XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss")
    xgb_params = {
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.9, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        xgb, xgb_params, n_iter=15, cv=3, scoring="f1_macro",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb_search.fit(X_train, y_train_enc)
    results["XGBoost"] = xgb_search.best_estimator_

    print("\n=== Classification Model Comparison ===")
    best_name, best_model, best_f1 = None, None, -1
    metrics_summary = {}
    for name, model in results.items():
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)
        f1 = f1_score(y_test_enc, preds, average="macro")
        try:
            auc = roc_auc_score(y_test_enc, proba, multi_class="ovr")
        except ValueError:
            auc = None
        print(f"\n--- {name} ---")
        print(classification_report(y_test_enc, preds, target_names=le.classes_))
        print(f"Macro F1: {f1:.3f} | ROC-AUC (OVR): {auc}")
        metrics_summary[name] = {"macro_f1": f1, "roc_auc_ovr": auc}
        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, model

    print(f"\nBest classification model: {best_name} (Macro F1={best_f1:.3f})")
    cm = confusion_matrix(y_test_enc, best_model.predict(X_test))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

    return best_name, best_model, le, metrics_summary


# ---------------------------------------------------------------------------
# Regression: remaining_shelf_life_hr
# ---------------------------------------------------------------------------
def train_regression(X_train, X_test, y_train, y_test):
    results = {}

    rf = RandomForestRegressor(random_state=RANDOM_STATE)
    rf_params = {
        "n_estimators": [200, 400, 600],
        "max_depth": [6, 10, 15, None],
        "min_samples_leaf": [1, 2, 4],
    }
    rf_search = RandomizedSearchCV(
        rf, rf_params, n_iter=15, cv=3, scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    results["RandomForest"] = rf_search.best_estimator_

    xgb = XGBRegressor(random_state=RANDOM_STATE)
    xgb_params = {
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.9, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        xgb, xgb_params, n_iter=15, cv=3, scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb_search.fit(X_train, y_train)
    results["XGBoost"] = xgb_search.best_estimator_

    print("\n=== Regression Model Comparison (remaining shelf life, hours) ===")
    best_name, best_model, best_rmse = None, None, np.inf
    metrics_summary = {}
    for name, model in results.items():
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"{name}: RMSE={rmse:.2f} hr | MAE={mae:.2f} hr | R2={r2:.3f}")
        metrics_summary[name] = {"rmse": rmse, "mae": mae, "r2": r2}
        if rmse < best_rmse:
            best_rmse, best_name, best_model = rmse, name, model

    print(f"\nBest regression model: {best_name} (RMSE={best_rmse:.2f} hr)")
    return best_name, best_model, metrics_summary


# ---------------------------------------------------------------------------
# SHAP Explainability
# ---------------------------------------------------------------------------
def generate_shap_summary(model, X_test, feature_names, out_dir):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        # Handle multiclass output across SHAP versions:
        # - list of per-class arrays (n_samples, n_features)
        # - single 3D array (n_samples, n_features, n_classes)
        # - single 2D array (n_samples, n_features)
        if isinstance(shap_values, list):
            mean_abs = np.mean([np.abs(v).mean(axis=0) for v in shap_values], axis=0)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            mean_abs = np.abs(shap_values).mean(axis=0).mean(axis=-1)
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)
        importance = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
        )
        importance.to_csv(out_dir / "shap_feature_importance.csv", index=False)
        print("\n=== Top 8 Features by SHAP Importance ===")
        print(importance.head(8).to_string(index=False))
        return explainer
    except Exception as e:
        print(f"SHAP generation skipped due to: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/raw/shipments.csv")
    parser.add_argument("--out_dir", type=str, default="../models/saved_models")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} shipment records.")

    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=df["spoilage_risk"]
    )
    df_feat, route_risk_map = engineer_features(df, train_idx=train_idx)

    X = build_feature_matrix(df_feat)
    y_clf = df_feat["spoilage_risk"]
    y_reg = df_feat["remaining_shelf_life_hr"]

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_clf_train, y_clf_test = y_clf.loc[train_idx], y_clf.loc[test_idx]
    y_reg_train, y_reg_test = y_reg.loc[train_idx], y_reg.loc[test_idx]

    clf_name, clf_model, le, clf_metrics = train_classification(
        X_train, X_test, y_clf_train, y_clf_test, list(X.columns)
    )
    reg_name, reg_model, reg_metrics = train_regression(
        X_train, X_test, y_reg_train, y_reg_test
    )

    explainer = generate_shap_summary(clf_model, X_test, list(X.columns), out_dir)

    # Save artifacts
    with open(out_dir / "classifier.pkl", "wb") as f:
        pickle.dump(clf_model, f)
    with open(out_dir / "regressor.pkl", "wb") as f:
        pickle.dump(reg_model, f)
    with open(out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open(out_dir / "route_risk_map.pkl", "wb") as f:
        pickle.dump(route_risk_map, f)
    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(list(X.columns), f)

    summary = {
        "best_classifier": clf_name,
        "classification_metrics": clf_metrics,
        "best_regressor": reg_name,
        "regression_metrics": reg_metrics,
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nAll artifacts saved to {out_dir.resolve()}")
    print("\n>>> Use these real numbers to fill in your resume bullets & interview answers.")


if __name__ == "__main__":
    main()
