import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

st.title('Supply Chain Risk & Delivery Time Prediction')
st.write('Enter the feature values to get predictions for risk classification and delivery time deviation.')

# Load the models and scalers
@st.cache_resource
def load_artifacts():
    with open('best_classification_model.pkl', 'rb') as f:
        clf_model = pickle.load(f)
    with open('best_regression_model.pkl', 'rb') as f:
        reg_model = pickle.load(f)
    with open('scaler_classification.pkl', 'rb') as f:
        scaler_clf = pickle.load(f)
    with open('scaler_regression.pkl', 'rb') as f:
        scaler_reg = pickle.load(f)
    
    # Re-initialize LabelEncoder for inverse_transform
    le = LabelEncoder()
    le.fit(['High Risk', 'Low Risk', 'Moderate Risk']) # Fit with known classes
    return clf_model, reg_model, scaler_clf, scaler_reg, le

clf_model, reg_model, scaler_clf, scaler_reg, le = load_artifacts()

# Define the input features based on the X DataFrame
# Make sure these columns match the order and names used during training
feature_columns = [
    'vehicle_gps_latitude', 'vehicle_gps_longitude', 'fuel_consumption_rate',
    'eta_variation_hours', 'traffic_congestion_level', 'warehouse_inventory_level',
    'loading_unloading_time', 'handling_equipment_availability', 'order_fulfillment_status',
    'weather_condition_severity', 'port_congestion_level', 'shipping_costs',
    'supplier_reliability_score', 'lead_time_days', 'historical_demand',
    'iot_temperature', 'cargo_condition_status', 'route_risk_level',
    'customs_clearance_time', 'driver_behavior_score', 'fatigue_monitoring_score',
    'disruption_likelihood_score', 'delay_probability'
]

# Create input widgets for each feature
input_data = {}
for col in feature_columns:
    # Using a generic number input; could customize based on feature's range/type
    input_data[col] = st.number_input(f'Enter {col}', value=0.0, step=0.01)

# Create a DataFrame from inputs
input_df = pd.DataFrame([input_data])

if st.button('Predict'):
    # Scale features using the respective scalers
    scaled_input_clf = scaler_clf.transform(input_df)
    scaled_input_reg = scaler_reg.transform(input_df)

    # Make classification prediction
    clf_prediction_encoded = clf_model.predict(scaled_input_clf)
    clf_prediction = le.inverse_transform(clf_prediction_encoded)

    # Make regression prediction
    reg_prediction = reg_model.predict(scaled_input_reg)

    st.subheader('Prediction Results:')
    st.write(f"Risk Classification: **{clf_prediction[0]}**")
    st.write(f"Delivery Time Deviation: **{reg_prediction[0]:.2f} hours**")

