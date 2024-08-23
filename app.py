import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# If your logo is stored locally in your repo, use a relative path
LOGO_URL_LARGE = "https://static.wixstatic.com/media/71ac90_328ecaed562c43a0ba2fb4ba8fe777aa~mv2.jpg/v1/crop/x_818,y_1584,w_2531,h_1013/fill/w_460,h_180,al_c,q_80,usm_0.66_1.00_0.01,enc_auto/name-change-2lvc-rgb.jpg" 
LOGO_LINK = "https://www.2l.vc/"

st.markdown(
    f"""
    <a href="{LOGO_LINK}" target="_blank">
        <img src="{LOGO_URL_LARGE}" alt="Logo" width="200">
    </a>
    """,
    unsafe_allow_html=True
)

# Step 1: Load the saved Random Forest model and feature names
model = joblib.load('logistic_regression_model.pkl')
feature_names = model.feature_names_in_  # Assuming you have this attribute stored or saved separately

# Step 2: Create a function to take user input and make predictions
def predict_user_input(input_data):
    # Align input_data with the feature names used during training
    input_data_aligned = input_data.reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(input_data_aligned)
    prediction_proba = model.predict_proba(input_data_aligned)
    return prediction, prediction_proba

# Step 3: Set up Streamlit interface
st.title('Predict whether a startup will M&A or raise Post-Seed')
st.header('By Anthony Khaiat')

# User input for the features
business_age_category_within_ten_years = st.number_input('Was the Business founded 5-10 years ago', min_value=0)
company_financing_formerly_vc_backed = st.number_input('Does the company have previous VC funding?', min_value=0)
is_in_new_york = st.number_input('Is the company based in New York City?', min_value=0)
is_in_software = st.number_input('Does the company develop software applications?', min_value=0)
is_in_san_francisco = st.number_input('Is the company based in San Francisco?', min_value=0)
is_in_everleigh = st.number_input('Is the company based in Everleigh?', min_value=0)
is_in_framingham = st.number_input('Is the company based in Framingham, MA?', min_value=0)
is_in_haifa = st.number_input('Is the company based in Haifa?', min_value=0)
is_in_devices = st.number_input('Does the company develop devices or electrical equipment?', min_value=0)
is_in_santa_clara = st.number_input('Is the company based in Santa Clara?', min_value=0)

# Combine user input into a single DataFrame
input_data = {
    'Business Age Category_Within 10 years': business_age_category_within_ten_years,
    'Company Financing Status_Formerly VC-backed': company_financing_formerly_vc_backed,
    'HQ City_New York': is_in_new_york,
    'Primary Industry Code_Software Development Applications': is_in_software,
    'HQ City_San Francisco': is_in_san_francisco,
    'HQ City_Everleigh': is_in_everleigh,
    'HQ City_Framingham': is_in_framingham,
    'HQ City_Haifa': is_in_haifa,
    'Primary Industry Code_Other Devices and Supplies': is_in_devices,
    'HQ City_Santa Clara': is_in_santa_clara
}

input_df = pd.DataFrame(input_data, index=[0])

# Step 4: Align input_df with the features used during model training
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Button to make predictions
if st.button('Predict'):
    prediction, prediction_proba = predict_user_input(input_df)
    if prediction[0] == 0:
        st.header(f'The model predicts the company WILL NOT raise at a later stage or M&A with {round(prediction_proba[0][0],2)*100}% confidence')
    else:
        st.header(f'The model predicts the company WILL raise at a later stage or M&A with {round(prediction_proba[0][1],2)*100}% confidence')

# Optional: Display the input data for confirmation
st.write('Input Data:')
st.write(input_df)
