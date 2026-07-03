"""
Streamlit app for interactive churn prediction.

Run:
    streamlit run app.py
"""
import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/churn_model.pkl"
ENCODERS_PATH = "models/encoders.pkl"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Predictor")
st.write("Enter a customer's details to estimate their probability of churning.")

if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH)):
    st.error("Model not found. Run `python src/train.py` first to train and save the model.")
    st.stop()

model = joblib.load(MODEL_PATH)
saved = joblib.load(ENCODERS_PATH)
encoders, columns = saved["encoders"], saved["columns"]

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    with col2:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        monthly_charges = st.slider("Monthly Charges ($)", 0.0, 150.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges * tenure))
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")

if submitted:
    row = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    df = pd.DataFrame([row])

    for col, le in encoders.items():
        if col in df.columns:
            val = str(df.at[0, col])
            df[col] = le.transform([val if val in le.classes_ else le.classes_[0]])

    df = df.reindex(columns=columns, fill_value=0)

    proba = model.predict_proba(df)[0, 1]
    pred = model.predict(df)[0]

    st.subheader("Result")
    if pred == 1:
        st.error(f"⚠️ Likely to churn — probability: {proba:.1%}")
    else:
        st.success(f"✅ Likely to stay — churn probability: {proba:.1%}")
    st.progress(min(int(proba * 100), 100))
