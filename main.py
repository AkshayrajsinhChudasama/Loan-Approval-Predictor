import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Changed to RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# --- THIS MUST BE THE VERY FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Loan Prediction App", layout="centered")

# --- 1. Load the pre-trained model and data (Alternative to direct download) ---

@st.cache_resource # Cache the model loading/training
def train_and_get_model_and_le():
    try:
        df = pd.read_csv('Dataset.csv') # Ensure this file is in the same directory
    except FileNotFoundError:
        st.error("Error: 'Dataset.csv' not found. Please ensure it's in the same directory as main.py")
        st.stop() # Stop the app if data is not found

    # Fill null values (as per your notebook)
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())

    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

    # Feature Engineering (as per your notebook)
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['ApplicantIncomelog'] = np.log(df['ApplicantIncome'] + 1)
    df['LoanAmountlog'] = np.log(df['LoanAmount'] + 1)
    df['Loan_Amount_Term_log'] = np.log(df['Loan_Amount_Term'] + 1)
    df['Total_Income_log'] = np.log(df['Total_Income'] + 1)

    # Drop unnecessary columns (as per your notebook)
    cols_to_drop = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Total_Income','Loan_ID']
    df = df.drop(columns = cols_to_drop, axis = 1)

    # Encoding Technique (as per your notebook)
    categorical_cols = ['Gender','Married','Education','Dependents','Self_Employed','Property_Area','Loan_Status']
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Split data (as per your notebook)
    X = df.drop(columns = ['Loan_Status'],axis = 1)
    y = df['Loan_Status']

    # Handle imbalance using RandomOverSampler (as per your notebook)
    from imblearn.over_sampling import RandomOverSampler
    oversample = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversample.fit_resample(X,y)

    # Train the model (Random Forest Classifier)
    # You can add parameters like n_estimators, max_depth, random_state etc.
    model = RandomForestClassifier(random_state=42) # Added random_state for reproducibility
    model.fit(X_resampled, y_resampled) # Train on the entire resampled data for deployment

    return model, le_dict, X.columns

model, le_dict, feature_columns = train_and_get_model_and_le()


# --- 2. Streamlit App Interface ---

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTitle {
        color: #2e86c1;
        text-align: center;
    }
    .stText {
        font-size: 18px;
    }
    .prediction-box {
        background-color: #e0f2f1;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .prediction-success {
        color: #28a745;
        font-size: 24px;
        font-weight: bold;
    }
    .prediction-failure {
        color: #dc3545;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💰 Loan Status Predictor")
st.write("Enter the applicant's details to predict their loan approval status.")

# --- 3. User Input Fields ---
st.header("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Gender", ("Male", "Female"))
    married = st.radio("Married", ("Yes", "No"))
    dependents = st.selectbox("Dependents", ("0", "1", "2", "3+"))
    education = st.radio("Education", ("Graduate", "Not Graduate"))
    self_employed = st.radio("Self Employed", ("Yes", "No"))

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=100)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0, step=100)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1, value=120, step=1) * 1000 # Convert to actual amount
    loan_amount_term = st.selectbox("Loan Amount Term (in months)", (12, 36, 60, 120, 180, 240, 300, 360, 480))
    credit_history = st.selectbox("Credit History", (0, 1), format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)")
    property_area = st.selectbox("Property Area", ("Urban", "Rural", "Semiurban"))

# --- 4. Preprocess User Input ---
def preprocess_input(gender, married, dependents, education, self_employed,
                     applicant_income, coapplicant_income, loan_amount,
                     loan_amount_term, credit_history, property_area, le_dict):

    # Create a dictionary for the input, matching the original DataFrame columns
    input_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'Credit_History': credit_history,
        'Property_Area': property_area,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Feature Engineering (must match training)
    input_df['Total_Income'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
    input_df['ApplicantIncomelog'] = np.log(input_df['ApplicantIncome'] + 1)
    input_df['LoanAmountlog'] = np.log(input_df['LoanAmount'] + 1)
    input_df['Loan_Amount_Term_log'] = np.log(input_df['Loan_Amount_Term'] + 1)
    input_df['Total_Income_log'] = np.log(input_df['Total_Income'] + 1)

    # Drop original columns that were dropped during training
    cols_to_drop_input = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Total_Income']
    input_df = input_df.drop(columns = cols_to_drop_input, axis = 1)


    # Apply Label Encoding (manual mapping as before)
    input_df['Gender'] = 1 if gender == 'Male' else 0 # Male:1, Female:0
    input_df['Married'] = 1 if married == 'Yes' else 0 # Yes:1, No:0
    input_df['Education'] = 0 if education == 'Graduate' else 1 # Graduate:0, Not Graduate:1
    input_df['Self_Employed'] = 1 if self_employed == 'Yes' else 0 # Yes:1, No:0

    if dependents == '3+':
        input_df['Dependents'] = 3
    else:
        input_df['Dependents'] = int(dependents)

    property_area_mapping = {'Semiurban': 2, 'Urban': 1, 'Rural': 0}
    input_df['Property_Area'] = property_area_mapping[property_area]

    # Ensure the order of columns matches the training data
    input_processed = input_df[feature_columns]

    return input_processed

# --- 5. Prediction and Output ---
if st.button("Predict Loan Status"):
    processed_input = preprocess_input(
        gender, married, dependents, education, self_employed,
        applicant_income, coapplicant_income, loan_amount,
        loan_amount_term, credit_history, property_area, le_dict
    )

    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)

    st.subheader("Prediction Result:")
    st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
    if prediction[0] == 1:
        st.markdown("<p class='prediction-success'>Loan Approved! 🎉</p>", unsafe_allow_html=True)
        st.write(f"Confidence (Approved): **{prediction_proba[0][1]*100:.2f}%**")
        st.write(f"Confidence (Not Approved): {prediction_proba[0][0]*100:.2f}%")
        st.balloons()
    else:
        st.markdown("<p class='prediction-failure'>Loan Not Approved. 😔</p>", unsafe_allow_html=True)
        st.write(f"Confidence (Not Approved): **{prediction_proba[0][0]*100:.2f}%**")
        st.write(f"Confidence (Approved): {prediction_proba[0][1]*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("---")
    st.info("Note: This prediction is based on a Random Forest Classifier model trained on the provided dataset.")