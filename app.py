import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ================= SHAP SETUP =================
shap.initjs()

# ================= LOAD TRAINED ARTIFACTS =================
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
num_cols = joblib.load("num_cols.pkl")
cat_cols = joblib.load("cat_cols.pkl")
feature_names = joblib.load("feature_names.pkl")
THRESHOLD = joblib.load("threshold.pkl")

# ================= SESSION STATE INIT =================
if "predicted" not in st.session_state:
    st.session_state.predicted = False
    st.session_state.prob = None
    st.session_state.X_final = None

# ================= PAGE SETUP =================
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

st.title("📉 Customer Churn Prediction System")
st.write(
    "This application predicts customer churn probability "
    "and explains the prediction using SHAP."
)

# ================= USER INPUT =================
st.header("🧾 Customer Information")

tenure = st.number_input(
    "Tenure in Months",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charge = st.number_input(
    "Monthly Charge",
    min_value=0.0,
    max_value=500.0,
    value=70.0
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One Year", "Two Year"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

# ================= BUILD INPUT DATA =================
input_data = {}

# Numerical features
for col in num_cols:
    if col == "Tenure in Months":
        input_data[col] = tenure
    elif col == "Monthly Charge":
        input_data[col] = monthly_charge
    else:
        input_data[col] = 0

# Categorical features
for col in cat_cols:
    if col == "Contract":
        input_data[col] = contract
    elif col == "Payment Method":
        input_data[col] = payment
    elif col in ["Internet Service", "Internet Type"]:
        input_data[col] = internet
    else:
        idx = list(cat_cols).index(col)
        input_data[col] = encoder.categories_[idx][0]

input_df = pd.DataFrame([input_data])

# ================= PREPROCESS =================
X_num = scaler.transform(input_df[num_cols])
X_cat = encoder.transform(input_df[cat_cols])

X_final = np.hstack([X_num, X_cat])
X_final = imputer.transform(X_final)

# ================= PREDICT BUTTON =================
if st.button("🔍 Predict Churn"):
    prob = model.predict_proba(X_final)[0, 1]

    st.session_state.predicted = True
    st.session_state.prob = prob
    st.session_state.X_final = X_final

# ================= SHOW RESULTS =================
if st.session_state.predicted:

    prob = st.session_state.prob
    X_final = st.session_state.X_final

    # Risk bucket
    if prob >= 0.7:
        risk = "High"
    elif prob >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    st.subheader("📊 Prediction Result")
    st.metric("Churn Probability", f"{prob:.2f}")

    if prob >= THRESHOLD:
        st.warning(f"⚠️ {risk} Risk of Churn (Action Required)")
        st.write("### ✅ Suggested Retention Actions")
        st.write("- Offer personalized discount")
        st.write("- Promote long-term contract")
        st.write("- Schedule retention call")
    else:
        st.success(f"✅ {risk} Risk (No Immediate Action Required)")

    # ================= SHAP EXPLAINABILITY =================
    st.markdown("---")

    if st.checkbox("🔍 Explain Prediction (SHAP)"):

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_final)

        # ----- SAME LOGIC AS YOUR JUPYTER NOTEBOOK -----
        if isinstance(shap_values, list):
            shap_single = shap_values[1][0]          # class 1 (churn)
            base_val = explainer.expected_value[1]
        else:
            shap_single = shap_values[0, :, 1]
            base_val = explainer.expected_value

        base_val = float(np.array(base_val).flatten()[0])

        st.subheader("📌 Why this customer may churn")

        single_exp = shap.Explanation(
            values=shap_single,
            base_values=base_val,
            data=X_final[0],
            feature_names=feature_names
        )

        fig = plt.figure()
        shap.plots.waterfall(single_exp, show=False)
        st.pyplot(fig)
        plt.clf()
