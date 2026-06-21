import streamlit as st
import joblib
import pandas as pd

# =========================
# LOAD FILES
# =========================

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Customer Churn Prediction")

st.markdown("""
This application predicts the probability of customer churn
using a Logistic Regression model trained on the Telco Customer
Churn dataset.
""")

st.write("Enter customer details below:")

# =========================
# USER INPUTS
# =========================

gender = st.selectbox("Gender", ["Female", "Male"])

SeniorCitizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

Partner = st.selectbox(
    "Partner",
    ["No", "Yes"]
)

Dependents = st.selectbox(
    "Dependents",
    ["No", "Yes"]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

PhoneService = st.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["No", "Yes"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["No", "Yes"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["No", "Yes"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["No", "Yes"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["No", "Yes"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes"]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

Contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

# =========================
# PREDICT BUTTON
# =========================

if st.button("Predict Churn"):

    input_data = {
        'gender': 1 if gender == 'Male' else 0,
        'SeniorCitizen': SeniorCitizen,
        'Partner': 1 if Partner == 'Yes' else 0,
        'Dependents': 1 if Dependents == 'Yes' else 0,
        'tenure': tenure,
        'PhoneService': 1 if PhoneService == 'Yes' else 0,
        'MultipleLines': 1 if MultipleLines == 'Yes' else 0,
        'OnlineSecurity': 1 if OnlineSecurity == 'Yes' else 0,
        'OnlineBackup': 1 if OnlineBackup == 'Yes' else 0,
        'DeviceProtection': 1 if DeviceProtection == 'Yes' else 0,
        'TechSupport': 1 if TechSupport == 'Yes' else 0,
        'StreamingTV': 1 if StreamingTV == 'Yes' else 0,
        'StreamingMovies': 1 if StreamingMovies == 'Yes' else 0,
        'PaperlessBilling': 1 if PaperlessBilling == 'Yes' else 0,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,

        'InternetService_Fiber optic':
            1 if InternetService == 'Fiber optic' else 0,

        'InternetService_No':
            1 if InternetService == 'No' else 0,

        'Contract_One year':
            1 if Contract == 'One year' else 0,

        'Contract_Two year':
            1 if Contract == 'Two year' else 0,

        'PaymentMethod_Credit card (automatic)':
            1 if PaymentMethod == 'Credit card (automatic)' else 0,

        'PaymentMethod_Electronic check':
            1 if PaymentMethod == 'Electronic check' else 0,

        'PaymentMethod_Mailed check':
            1 if PaymentMethod == 'Mailed check' else 0
    }

    input_df = pd.DataFrame([input_data])

    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"Customer likely to churn ({probability:.2%} probability)"
        )
    else:
        st.success(
            f"Customer likely to stay ({1-probability:.2%} probability)"
        )
     
    st.subheader("Risk Assessment")
    if probability >= 0.70:
        st.error("🔴 High Churn Risk")

    elif probability >= 0.40:
        st.warning("🟠 Medium Churn Risk")

    else:
        st.success("🟢 Low Churn Risk")

    st.metric(
        label="Churn Probability",
        value=f"{probability:.2%}"
    )
