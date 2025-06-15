import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PIL import Image
import os

st.markdown("""
<style>
.sidebar .sidebar-content {
    background-color: #f0f2f6;
}
.stButton>button {
    background-color: #4CAF50;
    color:white;
}
</style>
""", unsafe_allow_html=True)


st.markdown("<h3 style='text-align: left;'>Predictive models for neurosyphilis assessment in six global diagnosis guildances</h3>", unsafe_allow_html=True)


# Sidebar - Model selection
guidance_options = {
    "US 2018": "US2018_binary_nusvc",
    "US 2021": "US2021_binary_xgb",
    "Up to date": "Uptodate_svc",
    "Europe": "Euro_xgb",
    "China": "China_binary_rf",
    "Australia":"Aust_binary_xgb"
}

fig_options = {
    "US 2018": "US2018_binary nusvc",
    "US 2021": "US2021_binary xgb",
    "Up to date": "Uptodate svc",
    "Europe": "Euro xgb",
    "China": "China_binary rf",
    "Australia":"Aust_binary xgb"
}

fig2_options = {
    "US 2018": "US2018_binary",
    "US 2021": "US2021_binary",
    "Up to date": "Uptodate",
    "Europe": "Euro xgb",
    "China": "China_binary",
    "Australia":"Aust_binary"
}

st.sidebar.image("logo.png", width=200)

selected_guidance = st.sidebar.selectbox("Select Diagnosis Guildances:", list(guidance_options.keys()))


# --- Create two tabs ---
tab1, tab2, tab3, tab4, tab5= st.tabs(["✅ Prediction",'🔥 Shap values' ,"📈 ROC/AUC","🏥 Net benefit","👁️‍🗨️ Calibration"])

with tab1:
    st.markdown("<h4>Enter Patient Information</h4>", unsafe_allow_html=True)

    csf_pro = st.number_input("CSF Protein", min_value=0.0, max_value=1000.0, value=340.0)
    csf_wbc = st.number_input("CSF WBC", min_value=0.0, max_value=1000.0, value=9.0)

    binary_labels = [
        'Serum_NTT', 'CSF_TPPA', 'CSF_NTT','Neuro_Sympt'
    ]

    binary_values = {}
    for label in binary_labels:
        user_input = st.selectbox(f"{label}", ["Negative", "Positive"])
        binary_values[label] = 1 if user_input == "Positive" else 0

    input_data = {
        'CSF_Pro': csf_pro,
        'CSF_WBC': csf_wbc,
        **binary_values
    }

    input_df = pd.DataFrame([input_data])

    if st.button("Predict"):
        guidance_folder = guidance_options[selected_guidance]
        model_path = f"models/{guidance_folder}.pkl"

        try:
            model = joblib.load(model_path)
            prob = model.predict_proba(input_df)[0][1]
            st.write(f'Neurosyphilis for {selected_guidance}')
            st.metric(label="Predicted Probability", value=f"{prob:.2%}")

        except Exception as e:
            st.error(f"Error loading model or predicting: {e}")

with tab2:
    st.markdown("<h4>Shap values</h4>", unsafe_allow_html=True)
    
    guidance1 = fig_options[selected_guidance]
    
    fig_filename = f"{guidance1} shap values.png"

    fig_path = os.path.join("figures/", fig_filename)

    if os.path.exists(fig_path):
        st.image(Image.open(fig_path), caption=f"Shap values - {selected_guidance}")
    else:
        st.warning(f"No images found for {selected_guidance}.")

with tab3:
    st.markdown("<h4>ROC/AUC</h4>", unsafe_allow_html=True)
    
    guidance1 = fig_options[selected_guidance]
    
    fig_filename = f"{guidance1} auc.png"

    fig_path = os.path.join("figures/", fig_filename)

    if os.path.exists(fig_path):
        st.image(Image.open(fig_path), caption=f"ROCAUC - {selected_guidance}")
    else:
        st.warning(f"No images found for {selected_guidance}.")


with tab4:

    guidance1 = fig2_options[selected_guidance]
    fig_filename1 = f"figures/{guidance1} net benefit.png"
    st.image(fig_filename1, caption="Net benefit")

with tab5:
    guidance1 = fig2_options[selected_guidance]  
    fig_filename2 = f"figures/{guidance1} calibration.png"
    st.image(fig_filename2, caption="Calibration")


