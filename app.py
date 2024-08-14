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
model = joblib.load('random_forest_model.pkl ')
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
years_in_business = st.number_input('Years in Business', min_value=0)

# Use st.date_input to take a date input from the user
first_financing_date = st.date_input('First Financing Date')
today_date = datetime.today().date()
days_since_first_financing = (today_date - first_financing_date).days

# Use st.date_input for the last financing date and calculate days since last financing
last_financing_date = st.date_input('Last Financing Date')
days_since_last_financing = (today_date - last_financing_date).days

total_raised = st.number_input('Total Raised (in millions USD)', min_value=0.0, format="%.2f")
last_financing_size = st.number_input('Last Financing Size (in millions USD)', min_value=0.0, format="%.2f")
competitor_count = st.number_input('Number of Competitors', min_value=0)
ownership_status_acquired_merged = st.selectbox('Was the startup previously acquired or merged? (0 = No, 1 = Yes)', [0, 1])
most_recent_employment_year = st.number_input('How many employees did the startup have 1 year ago?', min_value=0)
number_of_employees = st.number_input('What is the current number of employees?', min_value=0)
third_most_recent_employment_year = st.number_input('How many employees did the startup have 3 years ago?', min_value=0)

# Combine user input into a single DataFrame
input_data = {
    'Years in Business': years_in_business,
    'Days Since First Financing': days_since_first_financing,
    'Total Raised': total_raised,
    'Last Financing Size': last_financing_size,
    'Competitor Count': competitor_count,
    'Ownership Status_Acquired/Merged (Operating Subsidiary)': ownership_status_acquired_merged,
    'Days Since Last Financing': days_since_last_financing,
    'Most Recent Employment Year': most_recent_employment_year,
    '# Employees': number_of_employees,
    'Third Most Recent Employment Year': third_most_recent_employment_year
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
    #st.write(f'Prediction: {prediction[0]}')
    #st.write(f'Probability of each class: {prediction_proba[0]}')

# Optional: Display the input data for confirmation
st.write('Input Data:')
st.write(input_df)
