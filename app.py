import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

model = joblib.load("fraud_detection_model.pkl")
features = joblib.load("features.pkl")


def calculate_features(data):

    data["amount_ratio"] = (
        data["amount"] /
        (data["previous_amount"] + 1)
    )

    data["amount_deviation"] = abs(
        data["amount"] -
        data["previous_amount"]
    )

    data["velocity_10min"] = (
        data["transactions_last_10min"] / 10
    )

    data["velocity_1hour"] = (
        data["transactions_last_1hour"] / 60
    )

    data["night_transaction"] = (
        (data["hour"] < 6) |
        (data["hour"] >= 23)
    ).astype(int)

    data["high_velocity"] = (
        data["transactions_last_10min"] > 5
    ).astype(int)

    data["high_amount"] = (
        data["amount"] > 10000
    ).astype(int)

    data["location_anomaly"] = (
        data["distance_from_last_transaction"] > 200
    ).astype(int)

    return data


def generate_reasons(t):

    reasons = []

    if t["amount"] > 10000:
        reasons.append("Unusually high transaction amount")

    if t["amount_ratio"] > 3:
        reasons.append(
            "Amount is much higher than previous transaction"
        )

    if t["is_new_device"] == 1:
        reasons.append("New device detected")

    if t["transactions_last_10min"] > 5:
        reasons.append("High transaction velocity")

    if t["transactions_last_1hour"] > 15:
        reasons.append("Unusual number of transactions")

    if t["failed_attempts"] > 3:
        reasons.append("Multiple failed attempts")

    if t["distance_from_last_transaction"] > 200:
        reasons.append("Unusual location change")

    if t["merchant_risk"] > 0.8:
        reasons.append("High-risk merchant")

    if t["location_risk"] > 0.8:
        reasons.append("High-risk location")

    if t["night_transaction"] == 1:
        reasons.append("Unusual transaction time")

    if not reasons:
        reasons.append("No major anomalies detected")

    return reasons


st.title("🛡️ AI Fraud Detection System")

st.markdown(
    "### Real-time transaction risk analysis"
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:

    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=1.0,
        value=1500.0
    )

    hour = st.slider(
        "Transaction Hour",
        0,
        23,
        14
    )

    account_age_days = st.number_input(
        "Account Age (days)",
        min_value=1,
        value=300
    )

with col2:

    transactions_last_10min = st.number_input(
        "Transactions - Last 10 min",
        min_value=0,
        value=1
    )

    transactions_last_1hour = st.number_input(
        "Transactions - Last 1 hour",
        min_value=0,
        value=4
    )

    failed_attempts = st.number_input(
        "Failed Attempts",
        min_value=0,
        value=0
    )

with col3:

    device_age_days = st.number_input(
        "Device Age (days)",
        min_value=1,
        value=100
    )

    is_new_device = st.selectbox(
        "New Device?",
        ["No", "Yes"]
    )

    previous_amount = st.number_input(
        "Previous Transaction Amount (₹)",
        min_value=1.0,
        value=1200.0
    )

st.divider()

col1, col2, col3 = st.columns(3)

with col1:

    distance = st.number_input(
        "Distance from Previous Transaction (km)",
        min_value=0.0,
        value=10.0
    )

with col2:

    merchant_risk = st.slider(
        "Merchant Risk",
        0.0,
        1.0,
        0.2
    )

with col3:

    location_risk = st.slider(
        "Location Risk",
        0.0,
        1.0,
        0.2
    )

predict = st.button(
    "🔍 Analyze Transaction",
    use_container_width=True
)

if predict:

    transaction = pd.DataFrame([{
        "amount": amount,
        "hour": hour,
        "account_age_days": account_age_days,
        "transactions_last_10min":
            transactions_last_10min,
        "transactions_last_1hour":
            transactions_last_1hour,
        "failed_attempts":
            failed_attempts,
        "device_age_days":
            device_age_days,
        "is_new_device":
            1 if is_new_device == "Yes" else 0,
        "distance_from_last_transaction":
            distance,
        "previous_amount":
            previous_amount,
        "merchant_risk":
            merchant_risk,
        "location_risk":
            location_risk
    }])

    transaction = calculate_features(
        transaction
    )

    X = transaction[features]

    probability = model.predict_proba(X)[0][1]

    risk_score = probability * 100

    if risk_score < 30:

        status = "LOW RISK"
        st.success("✅ LOW RISK")

    elif risk_score < 70:

        status = "MEDIUM RISK"
        st.warning("⚠️ MEDIUM RISK")

    else:

        status = "HIGH RISK"
        st.error("🚨 HIGH RISK")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

    with c2:

        st.metric(
            "Risk Score",
            f"{risk_score:.2f}/100"
        )

    with c3:

        st.metric(
            "Status",
            status
        )

    st.subheader("🔎 Risk Explanation")

    reasons = generate_reasons(
        transaction.iloc[0]
    )

    for reason in reasons:

        st.write(
            "⚠️ " + reason
        )
