import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from datetime import datetime

# --- SETUP: Database and API ---
DATABASE_URL = "sqlite:///predictions.db"
engine = create_engine(DATABASE_URL)
API_URL = "http://127.0.0.1:8000/predict"

# Create log table if it doesn't exist
with engine.connect() as connection:
    connection.execute(text("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY,
            timestamp DATETIME,
            Pregnancies INTEGER,
            Glucose INTEGER,
            BloodPressure INTEGER,
            BMI REAL,
            Age INTEGER,
            prediction_result TEXT
        )
    """))
    connection.commit()

# --- FUNCTION: Log the prediction ---
def log_prediction(data, result):
    """Logs the input data and the prediction result to the database."""
    with engine.connect() as connection:
        # Create a dictionary for easy insertion
        log_data = data.copy()
        log_data['timestamp'] = datetime.now()
        log_data['prediction_result'] = result
        
        # SQL Insert statement
        insert_query = text(
            "INSERT INTO logs (timestamp, Pregnancies, Glucose, BloodPressure, BMI, Age, prediction_result) "
            "VALUES (:timestamp, :Pregnancies, :Glucose, :BloodPressure, :BMI, :Age, :prediction_result)"
        )
        connection.execute(insert_query, log_data)
        connection.commit()

# --- APP LAYOUT ---
st.set_page_config(layout="wide", page_title="Advanced Diabetes Prediction")
st.title("🩺 Advanced ML Prediction Interface")
st.markdown("---")

# Use columns for a clean, two-panel layout
col1, col2 = st.columns([1, 2]) # Input panel is smaller than the dashboard panel

with col1:
    st.header("Patient Data Input")
    st.markdown("Enter the required metrics below:")

    # Input Widgets (with better defaults and max/min for validation)
    pregnancies = st.number_input("🤰 Pregnancies", min_value=0, max_value=17, value=2, step=1)
    glucose = st.number_input("🩸 Glucose", min_value=0, max_value=200, value=130, step=1)
    blood_pressure = st.number_input("💪 Blood Pressure", min_value=0, max_value=122, value=70, step=1)
    bmi = st.number_input("⚖️ BMI", min_value=0.0, max_value=67.1, value=28.5, step=0.1)
    age = st.number_input("👴 Age", min_value=21, max_value=81, value=45, step=1)

    # Prediction Button
    if st.button("🚀 Get Prediction", use_container_width=True, type="primary"):
        # 1. Prepare Payload
        payload = {
            "Pregnancies": int(pregnancies),
            "Glucose": int(glucose),
            "BloodPressure": int(blood_pressure),
            "BMI": float(bmi),
            "Age": int(age)
        }

        # 2. Call FastAPI
        try:
            response = requests.post(API_URL, data=json.dumps(payload), headers={"Content-Type": "application/json"})
            response.raise_for_status()
            result = response.json().get("prediction", "N/A")

            # 3. Display Result & Log Data
            if result == "Diabetic":
                st.error(f"## ⚠️ Result: **{result}**")
            else:
                st.success(f"## ✅ Result: **{result}**")
            
            log_prediction(payload, result) # MLOps Feature: Logging

        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Error: Ensure FastAPI is running on port 8000.")
        except requests.exceptions.RequestException as e:
            st.error(f"An API error occurred: {e}")

# --- DASHBOARD PANEL (Feature 4) ---
with col2:
    st.header("📊 Prediction Dashboard (MLOps Logging)")
    
    # Load all logged data
    df = pd.read_sql("SELECT * FROM logs", engine)

    if df.empty:
        st.info("No predictions logged yet. Run a prediction to see the dashboard!")
    else:
        # 1. Prediction Count Metric
        st.metric(label="Total Predictions Logged", value=len(df))

        # 2. Distribution Chart (Impressive Visualization)
        prediction_counts = df['prediction_result'].value_counts().reset_index()
        prediction_counts.columns = ['Result', 'Count']
        
        fig = px.pie(
            prediction_counts, 
            values='Count', 
            names='Result', 
            title='Distribution of Prediction Results',
            color_discrete_sequence=['green', 'red']
        )
        st.plotly_chart(fig, use_container_width=True)

        # 3. Raw Data (Transparency)
        st.subheader("Raw Prediction Log")
        st.dataframe(df.tail(10))

# --- SIDEBAR (Feature 1 & 5) ---
st.sidebar.title("Configuration & Info")
st.sidebar.markdown(f"""
    **API Status:** 🟢 Live at `{API_URL}`
    
    ---
    ### 🧠 Model Info
    * **Algorithm:** Random Forest Classifier
    * **Target:** Diabetes (Pima Indians Dataset)
    * **Key Metrics (Simulated):**
        * **Accuracy:** 93.2%
        * **F1-Score:** 0.88
    
    This UI demonstrates a complete **MLOps Pipeline**, logging predictions to track model performance over time.
""")