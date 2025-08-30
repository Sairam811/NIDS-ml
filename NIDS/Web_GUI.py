import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------
# Load models and scaler
# -------------------------
logreg = joblib.load('logistic_regression_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('minmax_scaler.pkl')

# Selected features (must match training order)
selected_features = [
    "duration", "src_bytes", "dst_bytes", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "same_srv_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_serror_rate", "dst_host_rerror_rate",
    "flag", "protocol_type", "logged_in", "service", "hot"
]

# Attack label mapping
attack_label_map = {
    '0': 'normal', '1': 'back', '2': 'buffer_overflow', '3': 'ftp_write',
    '4': 'guess_passwd', '6': 'ipsweep', '7': 'land', '8': 'loadmodule',
    '9': 'multihop', '10': 'neptune', '11': 'nmap', '12': 'perl', '14': 'pod',
    '15': 'portsweep', '16': 'rootkit', '17': 'satan', '18': 'smurf',
    '19': 'spy', '20': 'teardrop', '23': 'apache2', '25': 'udpstorm',
    '27': 'xsnoop', '30': 'snmpgetattack', '32': 'httptunnel', '34': 'mailbomb',
    '35': 'named'
}

# -------------------------
# Streamlit Modern Layout
# -------------------------
st.set_page_config(page_title="IDS Classifier", layout="wide")
st.title("🔐 Intrusion Detection System - NSL-KDD")
st.markdown("### Real-time classification of network activity using ML models")

# Sidebar layout
st.sidebar.header("⚙️ Select Configuration")
model_choice = st.sidebar.radio("🔧 Select Model", ["Logistic Regression", "XGBoost", "Random Forest", "SVM"])

st.sidebar.header("📋 Input Features")
sample = []
for feature in selected_features:
    value = st.sidebar.number_input(f"{feature}", value=0.0)
    sample.append(value)

# Main area layout
st.markdown("---")
st.subheader("🚀 Classification Result")
if st.sidebar.button("🔍 Predict"):
    sample_array = np.array(sample).reshape(1, -1)
    sample_scaled = scaler.transform(sample_array)

    if model_choice == "Logistic Regression":
        prediction = logreg.predict(sample_scaled)[0]
    elif model_choice == "XGBoost":
        prediction = xgb_model.predict(sample_scaled)[0]
    elif model_choice == "Random Forest":
        prediction = rf_model.predict(sample_scaled)[0]
    elif model_choice == "SVM":
        prediction = svm_model.predict(sample_scaled)[0]

    pred_label = str(prediction)
    pred_name = attack_label_map.get(pred_label, pred_label)
    st.success(f"✅ **Predicted Attack Type:** {pred_name}  ")
    st.code(f"Encoded Label: {pred_label}", language='text')
    
 # Visualization placeholder (simple bar)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    attack_names = list(attack_label_map.values())
    one_hot = [1 if k == pred_name else 0 for k in attack_names]
    ax.barh(attack_names, one_hot, color=["#ff4b4b" if k == pred_name else "#d3d3d3" for k in attack_names])
    ax.set_title("Predicted Class Highlight")
    ax.set_xlabel("Indicator")
    st.pyplot(fig)
 