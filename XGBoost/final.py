import streamlit as st
import joblib
import gdown
import os
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
# model = joblib.load("suicide_prediction_xgboost.pkl")
# vectorizer = joblib.load("tfidf_vectorizer.pkl")
# Google Drive File ID for model
# ------------------------------------------------------
file_id = "15oHrvU7ZOp1RKi-6Yu21XOHM2_V6pW6a"
url = f"https://drive.google.com/uc?id={file_id}"
output = "suicide_prediction_xgboost.pkl"

# Download the model if it doesn't exist
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# Load the model
model = joblib.load(output)
# ----------------------------------------------------------

file_id = "1xKe4Ox3KKR7Gx4eRbxZvOK0eJ-n7D-Jo"
url = f"https://drive.google.com/uc?id={file_id}"
output = "tfidf_vectorizer.pkl"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

vectorizer = joblib.load(output)

# ----------------------------------------------------------

# Streamlit UI Design
st.set_page_config(page_title="Suicide Prediction App", page_icon="🔍", layout="centered")

# Custom CSS for dark theme styling
st.markdown("""
    <style>
        body { background-color: #121212; color: white; }
        .main { background-color: #121212; }
        .stTextArea textarea { font-size: 16px; background-color: #333333; color: white; }
        .stButton button { background-color: #1E88E5; color: white; border-radius: 10px; padding: 10px 20px; font-size: 16px; }
        .stButton button:hover { background-color: #1565C0; }
        h1, h2, p { color: white; }
    </style>
""", unsafe_allow_html=True)

# App title with styling
st.markdown("<h1 style='text-align: center; color: #1E88E5;'>Suicide Prediction App</h1>", unsafe_allow_html=True)

st.write("**Enter a statement below to check if it indicates suicidal intent.**")

# Text input with placeholder
user_input = st.text_area("Enter your statement:", placeholder="Type your statement here...")

# Centered predict button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_btn = st.button("Predict")

if predict_btn:
    if user_input.strip():
        # Vectorize the input
        vectorized_input = vectorizer.transform([user_input])
        
        # Convert to DMatrix for XGBoost
        dmatrix_input = xgb.DMatrix(vectorized_input)
        
        # Make prediction
        prediction = model.predict(dmatrix_input)
        result = "Suicide" if prediction[0] >= 0.5 else "No Suicide"
        
        # Define messages
        if result == "Suicide":
            message = "<p style='color:red; font-size:18px; text-align:center;'><strong>Life is precious, and you matter. There are people who care about you. Please reach out for help.</strong></p>"
        else:
            message = "<p style='color:green; font-size:18px; text-align:center;'><strong>Stay strong! Every new day is a chance to find happiness and purpose.</strong></p>"
        
        # Display result with styled markdown
        st.markdown(f"<h2 style='text-align: center; color: {'red' if result == 'Suicide' else 'green'};'>Prediction: {result}</h2>", unsafe_allow_html=True)
        st.markdown(message, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a valid statement.")
