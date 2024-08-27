import streamlit as st
import joblib
import pandas as pd

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

# User input for the top 20 features
business_age_category_within_ten_years = st.selectbox('Was the business founded within the last 10 years?', options=[0, 1])
company_financing_formerly_vc_backed = st.selectbox('Is the company formerly VC-backed?', options=[0, 1])
hq_city_new_york = st.selectbox('Is the company based in New York?', options=[0, 1])
primary_industry_software_development = st.selectbox('Is the primary industry software development?', options=[0, 1])
hq_city_san_francisco = st.selectbox('Is the company based in San Francisco?', options=[0, 1])
most_recent_employment_year = st.number_input('How many employees did the startup have last year?', min_value=0)
hq_city_framingham = st.selectbox('Is the company based in Framingham?', options=[0, 1])
hq_city_everleigh = st.selectbox('Is the company based in Everleigh?', options=[0, 1])
primary_industry_apparel_accessories = st.selectbox('Is the primary industry apparel and accessories?', options=[0, 1])
primary_industry_clothing = st.selectbox('Is the primary industry clothing?', options=[0, 1])
primary_industry_other_devices_supplies = st.selectbox('Is the primary industry other devices and supplies?', options=[0, 1])
company_financing_formerly_angel_backed = st.selectbox('Is the company formerly angel-backed?', options=[0, 1])
primary_industry_commercial_services = st.selectbox('Is the primary industry commercial services?', options=[0, 1])
hq_city_haifa = st.selectbox('Is the company based in Haifa?', options=[0, 1])
hq_city_santa_clara = st.selectbox('Is the company based in Santa Clara?', options=[0, 1])
primary_industry_human_capital_services = st.selectbox('Is the primary industry human capital services?', options=[0, 1])
hq_city_saskatoon = st.selectbox('Is the company based in Saskatoon?', options=[0, 1])
hq_city_saint_louis = st.selectbox('Is the company based in Saint Louis?', options=[0, 1])
primary_industry_electronic_equipment_instruments = st.selectbox('Is the primary industry electronic equipment and instruments?', options=[0, 1])
hq_city_buffalo = st.selectbox('Is the company based in Buffalo?', options=[0, 1])

# Combine user input into a single DataFrame
input_data = {
    'Business Age Category_Within 10 years': business_age_category_within_ten_years,
    'Company Financing Status_Formerly VC-backed': company_financing_formerly_vc_backed,
    'HQ City_New York': hq_city_new_york,
    'Primary Industry Code_Software Development Applications': primary_industry_software_development,
    'HQ City_San Francisco': hq_city_san_francisco,
    'Most Recent Employment Year': most_recent_employment_year,
    'HQ City_Framingham': hq_city_framingham,
    'HQ City_Everleigh': hq_city_everleigh,
    'Primary Industry Group_Apparel and Accessories': primary_industry_apparel_accessories,
    'Primary Industry Code_Clothing': primary_industry_clothing,
    'Primary Industry Code_Other Devices and Supplies': primary_industry_other_devices_supplies,
    'Company Financing Status_Formerly Angel backed': company_financing_formerly_angel_backed,
    'Primary Industry Group_Commercial Services': primary_industry_commercial_services,
    'HQ City_Haifa': hq_city_haifa,
    'HQ City_Santa Clara': hq_city_santa_clara,
    'Primary Industry Code_Human Capital Services': primary_industry_human_capital_services,
    'HQ City_Saskatoon': hq_city_saskatoon,
    'HQ City_Saint Louis': hq_city_saint_louis,
    'Primary Industry Code_Electronic Equipment and Instruments': primary_industry_electronic_equipment_instruments,
    'HQ City_Buffalo': hq_city_buffalo
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
