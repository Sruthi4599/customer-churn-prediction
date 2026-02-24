import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ================= SHAP SETUP =================

# ================= LOAD ARTIFACTS =================
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
num_cols = joblib.load("num_cols.pkl")
cat_cols = joblib.load("cat_cols.pkl")
feature_names = joblib.load("feature_names.pkl")
THRESHOLD = joblib.load("threshold.pkl")

# ================= HUMAN LABELS =================
feature_labels = {
    "Contract_One Year": "contract duration",
    "Contract_Two Year": "long-term contract",
    "Internet Service_Fiber optic": "fiber internet service",
    "Internet Type_Fiber Optic": "fiber internet service",
    "Paperless Billing_Yes": "paperless billing usage",
    "Tenure in Months": "customer tenure",
    "Monthly Charge": "monthly subscription cost",
    "Payment Method_Electronic check": "payment method"
}

# ================= ACTIONS =================
feature_actions = {
    "Contract_One Year": "Offer discount for longer contracts",
    "Contract_Two Year": "Encourage multi-year subscription benefits",
    "Internet Service_Fiber optic": "Improve service quality assurance",
    "Payment Method_Electronic check": "Encourage automatic payments",
    "Tenure in Months": "Provide loyalty rewards"
}

# ================= SESSION =================
if "predicted" not in st.session_state:
    st.session_state.predicted = False
    st.session_state.prob = None
    st.session_state.X_final = None

# ================= PAGE =================
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📉 Customer Churn Prediction System")

st.write(
    "Predict customer churn probability and understand "
    "the business reasoning behind the prediction."
)

# ================= INPUT =================
st.header("Customer Information")

tenure = st.number_input("Tenure in Months", 0, 100, 12)
monthly_charge = st.number_input("Monthly Charge", 0.0, 500.0, 70.0)

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

# ================= BUILD INPUT =================
input_data = {}

for col in num_cols:
    if col == "Tenure in Months":
        input_data[col] = tenure
    elif col == "Monthly Charge":
        input_data[col] = monthly_charge
    else:
        input_data[col] = 0

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

# ================= PREDICT =================
if st.button("🔍 Predict Churn"):
    prob = model.predict_proba(X_final)[0, 1]

    st.session_state.predicted = True
    st.session_state.prob = prob
    st.session_state.X_final = X_final

# ================= RESULT =================
if st.session_state.predicted:

    prob = st.session_state.prob
    X_final = st.session_state.X_final

    if prob >= 0.7:
        risk = "High"
    elif prob >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{prob:.2f}")

    if prob >= THRESHOLD:
        st.warning(f"⚠️ {risk} Risk of Churn")
    else:
        st.success(f"✅ {risk} Risk")

    st.markdown("---")

    # ================= SHAP =================
    if st.checkbox("🔍 Explain Prediction"):

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_final)

        if isinstance(shap_values, list):
            shap_single = shap_values[1][0]
            base_val = explainer.expected_value[1]
        else:
            shap_single = shap_values[0, :, 1]
            base_val = explainer.expected_value

        base_val = float(np.array(base_val).flatten()[0])

        single_exp = shap.Explanation(
            values=shap_single,
            base_values=base_val,
            data=X_final[0],
            feature_names=feature_names
        )

        st.subheader("Model Explanation")

        fig = plt.figure()
        shap.plots.waterfall(single_exp, show=False)
        st.pyplot(fig)
        plt.clf()

        # ================= DATAFRAME =================
        shap_df = pd.DataFrame({
            "feature": feature_names,
            "impact": shap_single
        })

        top_features = shap_df.reindex(
            shap_df.impact.abs().sort_values(ascending=False).index
        ).head(3)

        # ================= WATERFALL STORY =================
        st.subheader("Prediction Summary")

        def readable(name):
            return feature_labels.get(
                name,
                name.replace("_", " ").lower()
            )

        increase_driver = shap_df.sort_values(
            by="impact", ascending=False
        ).iloc[0]

        decrease_driver = shap_df.sort_values(
            by="impact"
        ).iloc[0]

        summary = f"""
The model began with an average churn expectation and adjusted
the prediction based on customer characteristics.

Churn risk increased mainly due to **{readable(increase_driver['feature'])}**,
while **{readable(decrease_driver['feature'])}**
helped reduce the overall churn probability.

Overall, this customer falls under **{risk} churn risk**
with a predicted probability of **{prob:.2f}**.
"""

        st.info(summary)

        # ================= REASONS =================
        st.subheader("Key Drivers")

        for _, row in top_features.iterrows():

            label = readable(row["feature"])

            if row["impact"] > 0:
                st.write(
                    f"🔴 Customer's {label} contributed to higher predicted churn risk"
                )
            else:
                st.write(
                    f"🟢 Customer's {label} contributed to lower predicted churn risk"
                )

        # ================= ACTIONS =================
        st.subheader("Recommended Actions")

        actions = []

        for feature in top_features["feature"]:
            if feature in feature_actions:
                actions.append("✅ " + feature_actions[feature])
            else:
                actions.append(
                    f"✅ Review engagement related to {readable(feature)}"
                )

        for act in list(set(actions)):
            st.write(act)
