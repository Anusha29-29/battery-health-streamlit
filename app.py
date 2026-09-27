
import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Load trained model
# -----------------------------

model = joblib.load("battery_health_model.pkl")


# -----------------------------
# Load scaler if available
# -----------------------------

try:
    scaler = joblib.load("battery_health_scaler.pkl")
except:
    scaler = None


# -----------------------------
# Page title
# -----------------------------

st.title("Battery Health State Classification")

st.write(
    "Enter battery measurements to predict the battery health state."
)


# -----------------------------
# Input fields
# -----------------------------

Cycle = st.number_input(
    "Cycle",
    min_value=0.0,
    value=100.0
)

Voltage = st.number_input(
    "Voltage",
    value=3.7
)

Current = st.number_input(
    "Current",
    value=1.0
)

Temperature = st.number_input(
    "Temperature",
    value=25.0
)

ChargeTime = st.number_input(
    "Charge Time",
    value=2.0
)

DischargeTime = st.number_input(
    "Discharge Time",
    value=2.0
)

InternalResistance = st.number_input(
    "Internal Resistance",
    value=0.05
)

Capacity = st.number_input(
    "Capacity",
    value=2.0
)

AmbientHumidity = st.number_input(
    "Ambient Humidity",
    value=50.0
)

C_Rate = st.number_input(
    "C Rate",
    value=1.0
)


# -----------------------------
# Prediction button
# -----------------------------

if st.button("Predict Battery Health"):

    input_data = pd.DataFrame({
        "Cycle": [Cycle],
        "Voltage": [Voltage],
        "Current": [Current],
        "Temperature": [Temperature],
        "ChargeTime": [ChargeTime],
        "DischargeTime": [DischargeTime],
        "InternalResistance": [InternalResistance],
        "Capacity": [Capacity],
        "AmbientHumidity": [AmbientHumidity],
        "C_Rate": [C_Rate]
    })


    # -----------------------------
    # Apply scaler if required
    # -----------------------------

    if scaler is not None:
        input_data = scaler.transform(input_data)


    # -----------------------------
    # Prediction
    # -----------------------------

    prediction = model.predict(input_data)[0]


    # -----------------------------
    # Display result
    # -----------------------------

    st.success(
        f"Predicted Battery Health State: {prediction}"
    )


    # -----------------------------
    # Prediction probability
    # -----------------------------

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_data).max()

        st.write(
            f"Prediction Probability: {probability:.2%}"
        )
