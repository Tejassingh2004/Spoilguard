"""
SpoilGuard — Streamlit App
============================
Operational dashboard for predicting cold-chain shipment spoilage risk.

Run:
    streamlit run streamlit_app.py
(run this command from inside the app/ folder)
"""

import json
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st
from fpdf import FPDF

# Allow importing engineer_features/build_feature_matrix from src/
APP_DIR = Path(__file__).parent
sys.path.append(str(APP_DIR.parent / "src"))
from train_models import (  # noqa: E402
    engineer_features, build_feature_matrix, CATEGORICAL_COLS, NUMERIC_FEATURES,
)

MODEL_DIR = APP_DIR.parent / "models" / "saved_models"

# Product parameters (mirrors data_simulator.py — kept local so the app
# doesn't depend on the simulator module at runtime)
PRODUCTS = {
    "Milk & Dairy":       {"base_shelf_life_hr": 48,  "safe_temp_min": 2, "safe_temp_max": 8},
    "Leafy Greens":       {"base_shelf_life_hr": 96,  "safe_temp_min": 1, "safe_temp_max": 4},
    "Tomatoes & Produce": {"base_shelf_life_hr": 168, "safe_temp_min": 10, "safe_temp_max": 15},
    "Seafood":            {"base_shelf_life_hr": 36,  "safe_temp_min": 0, "safe_temp_max": 4},
    "Poultry & Meat":     {"base_shelf_life_hr": 72,  "safe_temp_min": 0, "safe_temp_max": 4},
    "Frozen Goods":       {"base_shelf_life_hr": 720, "safe_temp_min": -20, "safe_temp_max": -12},
    "Pharma (Vaccines)":  {"base_shelf_life_hr": 240, "safe_temp_min": 2, "safe_temp_max": 8},
    "Bakery":             {"base_shelf_life_hr": 48,  "safe_temp_min": 10, "safe_temp_max": 20},
}
VEHICLE_CONTROL_QUALITY = {
    "Refrigerated Truck": 0.9, "Insulated Non-Refrigerated": 0.5, "Standard Truck": 0.2,
}
PACKAGING_QUALITY_MAP = {
    "Premium (Insulated + Gel Packs)": 0.9, "Standard": 0.6, "Basic": 0.3,
}


@st.cache_resource
def load_artifacts():
    with open(MODEL_DIR / "classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(MODEL_DIR / "regressor.pkl", "rb") as f:
        reg = pickle.load(f)
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open(MODEL_DIR / "route_risk_map.pkl", "rb") as f:
        route_risk_map = pickle.load(f)
    with open(MODEL_DIR / "feature_columns.json") as f:
        feature_cols = json.load(f)
    return clf, reg, le, route_risk_map, feature_cols


def compute_raw_features(product, vehicle_type, packaging, distance_km,
                          avg_speed_kmph, ambient_temp_c, humidity_pct,
                          stop_overhead_hr=3.0):
    """Derives the raw shipment fields the model pipeline expects, from a
    small set of user-friendly inputs (mirrors the simulator's physics,
    minus the target-generating noise)."""
    p = PRODUCTS[product]
    transit_duration_hr = distance_km / avg_speed_kmph + stop_overhead_hr

    control_quality = VEHICLE_CONTROL_QUALITY[vehicle_type]
    pack_quality = PACKAGING_QUALITY_MAP[packaging]
    safe_mid = (p["safe_temp_min"] + p["safe_temp_max"]) / 2
    effective_temp = (
        ambient_temp_c * (1 - control_quality * pack_quality)
        + safe_mid * (control_quality * pack_quality)
    )

    if effective_temp > p["safe_temp_max"]:
        excursion_severity = effective_temp - p["safe_temp_max"]
    elif effective_temp < p["safe_temp_min"]:
        excursion_severity = p["safe_temp_min"] - effective_temp
    else:
        excursion_severity = 0.0
    temp_excursion_hours = excursion_severity * (transit_duration_hr / 10)

    return {
        "origin": "Custom", "destination": "Custom",
        "distance_km": distance_km,
        "product_category": product, "vehicle_type": vehicle_type,
        "packaging_quality": packaging,
        "transit_duration_hr": transit_duration_hr,
        "ambient_temp_c": ambient_temp_c, "humidity_pct": humidity_pct,
        "effective_transit_temp_c": effective_temp,
        "safe_temp_min_c": p["safe_temp_min"], "safe_temp_max_c": p["safe_temp_max"],
        "temp_excursion_hours": temp_excursion_hours,
        "base_shelf_life_hr": p["base_shelf_life_hr"],
    }


def predict(df_raw, clf, reg, le, route_risk_map, feature_cols):
    df_feat, _ = engineer_features(df_raw, route_risk_map=route_risk_map)
    X = build_feature_matrix(df_feat)
    # align columns with training-time feature set
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    risk_pred = le.inverse_transform(clf.predict(X))
    risk_proba = clf.predict_proba(X)
    shelf_life_pred = reg.predict(X)
    return risk_pred, risk_proba, shelf_life_pred, X


def make_pdf_report(results_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "SpoilGuard - Spoilage Risk Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {date.today().isoformat()}", ln=True)
    pdf.ln(4)

    high_risk_count = (results_df["spoilage_risk"] == "High").sum()
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, f"Total shipments: {len(results_df)} | High risk: {high_risk_count}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 9)
    headers = ["ID", "Product", "Risk", "Shelf Life (hr)"]
    widths = [20, 70, 40, 50]
    for h, w in zip(headers, widths):
        pdf.cell(w, 8, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for i, row in results_df.head(50).iterrows():
        pdf.cell(20, 7, str(i + 1), border=1)
        pdf.cell(70, 7, str(row.get("product_category", ""))[:35], border=1)
        pdf.cell(40, 7, str(row["spoilage_risk"]), border=1)
        pdf.cell(50, 7, f"{row['predicted_remaining_shelf_life_hr']:.1f}", border=1)
        pdf.ln()

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SpoilGuard", page_icon="🧊", layout="wide")
st.title("🧊 SpoilGuard — Cold Chain Spoilage Risk Predictor")
st.caption("Predicts spoilage risk & remaining shelf life for perishable shipments in transit.")

try:
    clf, reg, le, route_risk_map, feature_cols = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Run `python train_models.py` from the "
        "`src/` folder first to generate models/saved_models/*.pkl"
    )
    st.stop()

tab1, tab2 = st.tabs(["📦 Single Shipment", "📄 Batch Upload (CSV)"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        product = st.selectbox("Product Category", list(PRODUCTS.keys()))
        vehicle_type = st.selectbox("Vehicle Type", list(VEHICLE_CONTROL_QUALITY.keys()))
        packaging = st.selectbox("Packaging Quality", list(PACKAGING_QUALITY_MAP.keys()))
        distance_km = st.number_input("Distance (km)", min_value=1, value=500)
    with col2:
        avg_speed_kmph = st.number_input("Avg Speed (km/h)", min_value=10, value=45)
        ambient_temp_c = st.number_input("Ambient Temperature (°C)", value=28.0)
        humidity_pct = st.number_input("Humidity (%)", min_value=0, max_value=100, value=60)

    if st.button("Predict Risk", type="primary"):
        raw = compute_raw_features(
            product, vehicle_type, packaging, distance_km,
            avg_speed_kmph, ambient_temp_c, humidity_pct,
        )
        df_raw = pd.DataFrame([raw])
        risk_pred, risk_proba, shelf_life_pred, X = predict(
            df_raw, clf, reg, le, route_risk_map, feature_cols
        )

        risk = risk_pred[0]
        color = {"Low": "green", "Medium": "orange", "High": "red"}[risk]
        st.markdown(f"### Spoilage Risk: :{color}[{risk}]")
        st.metric("Estimated Remaining Shelf Life", f"{shelf_life_pred[0]:.1f} hours")

        if risk == "High":
            st.error("⚠️ ALERT: High spoilage risk — recommend expedited delivery or nearest-hub rerouting.")
        elif risk == "Medium":
            st.warning("⚠️ Monitor closely — approaching risk threshold.")
        else:
            st.success("✅ Shipment within safe parameters.")

        st.subheader("Why this prediction? (SHAP)")
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)
        class_idx = list(le.classes_).index(risk)
        if isinstance(shap_values, list):
            vals = shap_values[class_idx][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            vals = shap_values[0, :, class_idx]
        else:
            vals = shap_values[0]
        shap_df = pd.DataFrame(
            {"feature": X.columns, "impact": vals}
        ).sort_values("impact", key=abs, ascending=False).head(6)
        st.bar_chart(shap_df.set_index("feature")["impact"])

with tab2:
    st.write(
        "Upload a CSV with columns: `origin, destination, distance_km, "
        "product_category, vehicle_type, packaging_quality, transit_duration_hr, "
        "ambient_temp_c, humidity_pct, effective_transit_temp_c, safe_temp_min_c, "
        "safe_temp_max_c, temp_excursion_hours, base_shelf_life_hr`"
    )
    st.caption("Tip: rows from your `data/raw/shipments.csv` (minus the target columns) work directly.")

    uploaded = st.file_uploader("Upload shipment CSV", type="csv")
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        risk_pred, risk_proba, shelf_life_pred, X = predict(
            df_raw, clf, reg, le, route_risk_map, feature_cols
        )
        results = df_raw.copy()
        results["spoilage_risk"] = risk_pred
        results["predicted_remaining_shelf_life_hr"] = shelf_life_pred.round(1)

        n_high = (results["spoilage_risk"] == "High").sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Shipments", len(results))
        c2.metric("High Risk", n_high)
        c3.metric("% High Risk", f"{n_high / len(results) * 100:.1f}%")

        st.subheader("🚨 High-Risk Alerts")
        st.dataframe(
            results[results["spoilage_risk"] == "High"]
            .sort_values("predicted_remaining_shelf_life_hr")
        )

        st.subheader("All Predictions")
        st.dataframe(results)

        st.download_button(
            "⬇️ Download Predictions (CSV)",
            results.to_csv(index=False).encode(),
            "spoilguard_predictions.csv",
        )
        st.download_button(
            "⬇️ Download PDF Report",
            make_pdf_report(results),
            "spoilguard_report.pdf",
        )
