import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from sklearn.preprocessing import LabelEncoder

# --- Page Configuration ---
st.set_page_config(
    page_title="Loan Approval Predictor",
    layout="centered",
    page_icon="🏦"
)

# --- Custom Styling ---
st.markdown("""
    <style>
        .stApp {
            background-image: url("https://raw.githubusercontent.com/leonadsouza28/LOAN-APPROVAL-PREDICTION/refs/heads/main/LP_image.png");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .main {
            background-color: #F4F6F6;
            padding: 20px;
            border-radius: 10px;
        }
        .center-button {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .stButton>button {
            color: white;
            background-color: #0066cc;
            border-radius: 8px;
            padding: 12px 30px;
            font-size: 18px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #004d99;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #003366;
            text-align: center;
        }
        .subtitle {
            font-size: 18px;
            color: white;
            font-weight: bold;
        }
        /* 🔵 Make input boxes blue */
        div[data-baseweb="select"] > div {
            background-color: #dbe9ff !important;
            border: 1px solid #0066cc !important;
            border-radius: 6px !important;
        }
        input[type="number"], input[type="text"] {
            background-color: #dbe9ff !important;
            border: 1px solid #0066cc !important;
            border-radius: 6px !important;
        }
        /* ⚪ Make all selectbox/number input labels bold */
        .stSelectbox label, .stNumberInput label {
            color: black !important;
            font-weight: bold;
        }
        /* 🌑 Darken prediction boxes */
        div["data-testid="stSuccess"], div["data-testid="stError"] {
            background-color: rgba(0, 0, 0, 0.8) !important;
            color: white !important;
            border: 1px solid white !important;
            border-radius: 8px !important;
            padding: 15px !important;
            margin-top: 10px !important;
        }
        div["data-testid="stSuccess"] * , div["data-testid="stError"] * {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load Model & Preprocessing Files ---
model = joblib.load("loan_approval_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

# --- Title & Subtitle ---
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">LOAN APPROVAL PREDICTION</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict whether a loan will be approved based on applicant information.</div>', unsafe_allow_html=True)

# --- Input Form ---
with st.form(key="input_form"):
    col1, col2 = st.columns(2)

    with col1:
        Gender = st.selectbox("Gender", ['', 'Male', 'Female', 'Other'])
        Married = st.selectbox("Married", ['', 'Yes', 'No'])
        Dependents = st.selectbox("Number of Dependents", ['', '0', '1', '2', '3+'])
        Education = st.selectbox("Education", ['', 'Graduate', 'Not Graduate'])
        Self_Employed = st.selectbox("Self Employed", ['', 'Yes', 'No'])

    with col2:
        ApplicantIncome = st.number_input("Applicant Income", min_value=0)
        CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
        LoanAmount = st.number_input("Loan Amount (in Rupees)", min_value=0, step=1000)
        # Convert to thousands for the model
        LoanAmount_thousands = LoanAmount / 1000

        Loan_Amount_Term = st.selectbox("Loan Term (in years)", [''] + list(range(1, 1001)))
        Credit_History = st.selectbox("Credit History", ['', '1', '0'])
        Property_Area = st.selectbox("Property Area", ['', 'Urban', 'Semiurban', 'Rural'])

    # Submit button
    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    predict = st.form_submit_button("Predict Loan Approval")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction Logic ---
if predict:
    # Ensure no missing required fields
    required_fields = [Gender, Married, Dependents, Education, Self_Employed, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]
    if any(f == '' for f in required_fields):
        st.warning("⚠️ Please fill in all fields before predicting.")
    else:
        # Custom Loan Amount Validation
        if int(LoanAmount) < 1000:
            st.warning("⚠️ Loan Amount must be at least 1,000.")
        elif int(LoanAmount) <= (ApplicantIncome + CoapplicantIncome):
            st.warning("⚠️ Loan Amount must be greater than the total income (Applicant + Coapplicant).")
        else:
            # Step 1: Create raw input DataFrame
            input_dict = {
                'Gender': [Gender],
                'Married': [Married],
                'Dependents': [Dependents],
                'Education': [Education],
                'Self_Employed': [Self_Employed],
                'ApplicantIncome': [ApplicantIncome],
                'CoapplicantIncome': [CoapplicantIncome],
                'LoanAmount': [LoanAmount_thousands],
                'Loan_Amount_Term': [int(Loan_Amount_Term)],
                'Credit_History': [int(Credit_History)],
                'Property_Area': [Property_Area]
            }
            input_df = pd.DataFrame(input_dict)

            # Step 2: Encode categorical binary columns
            le = LabelEncoder()
            for col in ['Gender', 'Married', 'Education', 'Self_Employed']:
                input_df[col] = le.fit_transform(input_df[col])

            # Step 3: One-hot encode multi-class features
            input_df = pd.get_dummies(input_df, columns=['Dependents', 'Property_Area'], drop_first=True)

            # Step 4: Align with training columns
            input_df = input_df.reindex(columns=model_columns, fill_value=0)

            # Step 5: Scale input
            input_scaled = scaler.transform(input_df)

            # Get probability of prediction
            prediction_proba = model.predict_proba(input_scaled)

            # Probability of being approved (class = 1)
            confidence = prediction_proba[0][1] * 100  

            # Prevent showing exact 100.00%
            if confidence == 100:
                confidence = 99.99

            # Get prediction (0 = Not Approved, 1 = Approved)
            prediction = model.predict(input_scaled)[0]    

            # Show prediction
            if prediction == 1:
                st.success(f"✅ Loan will be Approved (Confidence: {confidence:.4f}%)")
            else:
                st.error(f"❌ Loan will NOT be Approved (Confidence: {100 - confidence:.4f}%)")
