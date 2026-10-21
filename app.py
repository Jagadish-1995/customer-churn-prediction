
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "customer_churn_model.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "churn_threshold.json"


# =========================================================
# 3. LOAD MODEL AND THRESHOLD
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_threshold():
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    return float(data["threshold"])


# =========================================================
# 4. VALIDATE REQUIRED FILES
# =========================================================

if not MODEL_PATH.is_file():
    st.error(
        f"Model file not found: {MODEL_PATH}\n\n"
        "Check that customer_churn_model.pkl is inside the models folder."
    )
    st.stop()

if not THRESHOLD_PATH.is_file():
    st.error(
        f"Threshold file not found: {THRESHOLD_PATH}\n\n"
        "Check that churn_threshold.json is inside the models folder."
    )
    st.stop()


try:
    model = load_model()
    threshold = load_threshold()

except Exception as error:
    st.error(f"Unable to load the model files: {error}")
    st.stop()


if not 0 <= threshold <= 1:
    st.error("The saved classification threshold must be between 0 and 1.")
    st.stop()


# =========================================================
# 5. APPLICATION HEADER
# =========================================================

st.title("Customer Churn Prediction")

st.write(
    "An end-to-end Machine Learning application that estimates "
    "the probability of a telecom customer leaving the service."
)

st.caption(
    "Model: Tuned AdaBoost | "
    f"Classification threshold: {threshold:.1%}"
)

with st.sidebar:
    st.header("About this project")

    st.write(
        "This application uses a trained Machine Learning pipeline "
        "to estimate customer churn risk from account and service details."
    )

    st.divider()

    st.subheader("Model information")

    st.write("Algorithm: AdaBoost Classifier")
    st.write(f"Decision threshold: {threshold:.2f}")

    st.info(
        "The threshold determines how probabilities are converted "
        "into churn and non-churn predictions."
    )

    st.caption(
        "Portfolio demonstration. Predictions are estimates, "
        "not guarantees."
    )


# =========================================================
# 6. CUSTOMER INPUT FORM
# =========================================================

st.header("Customer Information")

st.write(
    "Enter the customer's account and service details below."
)

with st.form("customer_form"):

    st.subheader("Personal and Account Details")

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No"
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

    with col2:

        tenure = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col3:

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )


    st.divider()

    st.subheader("Internet and Additional Services")

    service_col1, service_col2, service_col3 = st.columns(3)

    with service_col1:

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with service_col2:

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

    with service_col3:

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )


    st.divider()

    st.subheader("Billing Information")

    charge_col1, charge_col2 = st.columns(2)

    with charge_col1:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=1000.0,
            value=70.0,
            step=1.0
        )

    with charge_col2:

        calculate_total = st.checkbox(
            "Estimate total charges as monthly charges × tenure",
            value=False
        )

        total_charges_input = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=100000.0,
            value=840.0,
            step=10.0,
            disabled=calculate_total
        )


    st.caption(
        "If you enable automatic calculation, total charges will be "
        "estimated from the entered monthly charges and tenure. "
        "For a real customer, use their actual billing total whenever available."
    )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        type="primary",
        use_container_width=True
    )


# =========================================================
# 7. PREPARE CUSTOMER DATA
# =========================================================

if submitted:

    # Use a consistent total-charge calculation when requested.
    if calculate_total:
        total_charges = monthly_charges * tenure
    else:
        total_charges = total_charges_input


    # Normalise service categories to match the training data.
    # These rules prevent contradictory service combinations.

    if phone_service == "No":
        multiple_lines = "No phone service"

    if internet_service == "No":

        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"


    # Build a single-row DataFrame with the training feature names.
    customer = pd.DataFrame([{

        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges)

    }])


    # =====================================================
    # 8. VALIDATE MODEL INPUT SCHEMA
    # =====================================================

    expected_features = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges"
    ]

    if list(customer.columns) != expected_features:
        st.error("Customer feature columns do not match the expected schema.")
        st.stop()


    # =====================================================
    # 9. GENERATE PREDICTION
    # =====================================================

    try:

        # Check that class 1 represents churn in the saved model.
        model_classes = list(model.classes_)

        if 1 not in model_classes:
            st.error(
                "The saved model does not have class 1. "
                "Check how the Churn target was encoded during training."
            )
            st.stop()

        churn_class_index = model_classes.index(1)

        probabilities = model.predict_proba(customer)[0]

        churn_probability = float(
            probabilities[churn_class_index]
        )

        churn_prediction = int(
            churn_probability >= threshold
        )

    except Exception as error:

        st.error(
            "Prediction failed. Verify that the saved pipeline uses "
            "the same input features and compatible library versions."
        )

        st.exception(error)
        st.stop()


    # =====================================================
    # 10. DISPLAY CUSTOMER SUMMARY
    # =====================================================

    st.divider()

    st.header("Prediction Results")

    st.subheader("Customer Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric(
            "Customer Tenure",
            f"{tenure} months"
        )

    with summary_col2:
        st.metric(
            "Monthly Charges",
            f"${monthly_charges:,.2f}"
        )

    with summary_col3:
        st.metric(
            "Total Charges",
            f"${total_charges:,.2f}"
        )


    # =====================================================
    # 11. DISPLAY CHURN PROBABILITY AND CLASSIFICATION
    # =====================================================

    st.subheader("Churn Risk Assessment")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Estimated Churn Probability",
            f"{churn_probability:.1%}"
        )

    with result_col2:

        st.metric(
            "Classification Threshold",
            f"{threshold:.1%}"
        )


    st.progress(churn_probability)


    if churn_prediction == 1:

        st.error(
            "Potential Churn Risk",
            icon="⚠️"
        )

        st.write(
            "The estimated probability meets or exceeds the selected "
            "threshold. The customer may be considered for a retention "
            "review."
        )

    else:
        st.success(
        "Not Flagged as Potential Churn",
        icon="✅"
        )

        st.write(
        "The estimated probability is below the "
        "selected classification threshold."
        )


    # =====================================================
    # 12. DISPLAY CUSTOMER INPUTS
    # =====================================================

    with st.expander("View all model input features"):

        st.dataframe(
            customer,
            use_container_width=True,
            hide_index=True
        )


    # =====================================================
    # 13. MODEL LIMITATIONS
    # =====================================================

    st.divider()

    st.caption(
        "This application provides a statistical estimate, not a guarantee "
        "of future customer behaviour. Predictions depend on the training "
        "data, feature quality, model calibration and classification "
        "threshold. The threshold affects the trade-off between missed "
        "churners and false alarms."
    )
