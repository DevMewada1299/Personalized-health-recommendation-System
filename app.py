import streamlit as st

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Health Data Analysis",
    page_icon="🏥",
    layout="wide"
)

import pandas as pd
from typing import List, Dict, Any
from pymongo import MongoClient
import pymongo
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import os
import numpy as np
from datetime import datetime
import plotly.express as px
from clustering import predict_cluster
import plotly.graph_objects as go
from llm_utils import chat_with_gemini, chat_with_gemini_with_context
from pymongo import MongoClient
from bson import ObjectId
import json
from PyPDF2 import PdfReader

# Import local modules
import pdf_utils
import clustering
import llm_utils

def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Calculate BMI from weight in kg and height in cm."""
    if height_cm <= 0 or weight_kg <= 0:
        return 0
    height_m = height_cm / 100
    return round(weight_kg / (height_m * height_m), 1)

def get_bmi_category(bmi: float) -> str:
    """Return BMI category based on value."""
    if bmi <= 0:
        return "Unknown"
    elif bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def get_bp_category(systolic: int, diastolic: int) -> str:
    """Return blood pressure category based on values."""
    if systolic <= 0 or diastolic <= 0:
        return "Unknown"
    elif systolic < 120 and diastolic < 80:
        return "Normal"
    elif systolic < 130 and diastolic < 80:
        return "Elevated"
    elif (systolic < 140 and diastolic < 90) and (systolic >= 130 or diastolic >= 80):
        return "Hypertension Stage 1"
    else:
        return "Hypertension Stage 2"

def map_bp_to_csv_category(bp_category: str) -> str:
    """Map BP category to match dataset categories."""
    if "Normal" in bp_category:
        return "Normal"
    elif "Elevated" in bp_category:
        return "Elevated"
    elif "Hypertension" in bp_category:
        return "High"
    else:
        return "Normal"

def identify_risk_factors(user_data: Dict[str, Any]) -> List[str]:
    """Identify risk factors based on user data."""
    risks = []
    
    # BMI-related risks
    if user_data.get("bmi_category") == "Underweight":
        risks.append("Underweight")
    elif user_data.get("bmi_category") == "Overweight":
        risks.append("Overweight")
    elif user_data.get("bmi_category") == "Obese":
        risks.append("Obesity")
    
    # Blood pressure risks
    if "Hypertension" in user_data.get("bp_category", ""):
        risks.append("High blood pressure")
    
    # Heart rate risks
    if user_data.get("resting_hr", 0) > 100:
        risks.append("Elevated resting heart rate")
    
    # Lifestyle risks
    if user_data.get("smoking_status") == "Current smoker":
        risks.append("Smoking")
    
    if user_data.get("exercise_days", 0) < 3:
        risks.append("Insufficient physical activity")
    
    # Chronic conditions
    for condition in user_data.get("chronic_conditions", []):
        if condition != "None":
            risks.append(condition)
    
    # Symptoms
    for symptom in user_data.get("symptoms", []):
        risks.append(symptom)
    
    return risks

def create_user_account(name, email, age, gender, ethnicity, weight, height, bmi, bmi_category, cholesterol, systolic_bp, diastolic_bp, bp_category, resting_hr, smoking_status, exercise_days, diet, symptoms, chronic_conditions, password):
    user_data = {
        "name": name,
        "email": email,
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "weight_kg": weight,
        "height_cm": height,
        "bmi": bmi,
        "bmi_category": bmi_category,
        "cholesterol": cholesterol,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "bp_category": bp_category,
        "resting_hr": resting_hr,
        "smoking_status": smoking_status,
        "exercise_days": exercise_days,
        "diet": diet,
        "symptoms": symptoms if "None" not in symptoms else [],
        "chronic_conditions": chronic_conditions if "None" not in chronic_conditions else [],
        "password": password
    }
    db = connect_to_mongodb()
    result = db.users.insert_one(user_data)
    user_data["_id"] = str(result.inserted_id)
    st.session_state['user_account'] = user_data
    st.success("Account created successfully!")

def login_user(username, password):
    """Simulate user login by checking if the provided username and password match the stored user account in the database."""
    db = connect_to_mongodb()
    user = db.users.find_one({"email": username, "password": password})
    if user:
        st.success("Login successful!")
        user['_id'] = str(user['_id'])  # Convert ObjectId to string
        st.session_state['user_account'] = user
        return True
    else:
        st.error("Invalid username or password.")
        return False

def save_user_data_to_db(user_data):
    """Save user data to MongoDB."""
    db = connect_to_mongodb()
    try:
        db.users.insert_one(user_data)
        st.success("Data saved to database successfully!")
    except pymongo.errors.DuplicateKeyError:
        st.error("A user with this ID already exists. Please try again with a different ID.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def view_database():
    """Display the stored user data from MongoDB in a table format."""
    db = connect_to_mongodb()
    users = list(db.users.find())
    if db is not None and users:
        st.subheader("Stored User Data")
        df = pd.DataFrame(users)
        st.dataframe(df)
    else:
        st.info("No data stored in the database yet.")

def connect_to_mongodb():
    """Connect to MongoDB and return the database object."""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['health_app']
        return db
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {str(e)}")
        return None

# Admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Admin authentication
def admin_login():
    if not st.session_state.get('user_account') and not st.session_state.get('admin_logged_in'):
        st.sidebar.title("Admin Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login", key="admin_login_button"):
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state['admin_logged_in'] = True
                st.sidebar.success("Admin login successful!")
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials!")

# Admin panel
def show_admin_panel():
    st.title("Admin Panel")
    
    # Get all users from MongoDB
    db = connect_to_mongodb()
    if db is not None:
        users = list(db.users.find())
        
        # Convert ObjectId to string for display
        for user in users:
            user['_id'] = str(user['_id'])

        # --- Platform Health Summary (Aggregate Metrics and Visuals) ---
        st.subheader("📈 Platform Health Summary")

        # Compute averages
        avg_bmi = np.mean([u.get("bmi", 0) for u in users if u.get("bmi", 0) > 0])
        avg_age = np.mean([u.get("age", 0) for u in users if u.get("age", 0) > 0])
        avg_hr = np.mean([u.get("resting_hr", 0) for u in users if u.get("resting_hr", 0) > 0])
        st.metric("Average BMI", f"{avg_bmi:.1f}")
        st.metric("Average Age", f"{avg_age:.1f}")
        st.metric("Average Heart Rate", f"{avg_hr:.1f} bpm")

        # BMI distribution
        st.subheader("BMI Distribution")
        bmi_vals = [u.get("bmi", 0) for u in users if u.get("bmi", 0) > 0]
        st.plotly_chart(px.histogram(x=bmi_vals, nbins=20, labels={"x": "BMI"}, title="Histogram of BMI"))

        # Age distribution
        st.subheader("Age Distribution")
        age_vals = [u.get("age", 0) for u in users if u.get("age", 0) > 0]
        st.plotly_chart(px.histogram(x=age_vals, nbins=15, labels={"x": "Age"}, title="Age Distribution"))

        # Risk category pie chart
        st.subheader("Risk Category Breakdown")
        all_risks = []
        for u in users:
            df_risk = {
                "Age": u.get("age", 0),
                "Gender": u.get("gender", ""),
                "Blood Pressure": u.get("bp_category", "Normal"),
                "Cholesterol Level": u.get("cholesterol", "Normal"),
                "Fever": 1 if "Fever" in u.get("symptoms", []) else 0,
                "Cough": 1 if "Cough" in u.get("symptoms", []) else 0,
                "Fatigue": 1 if "Fatigue" in u.get("symptoms", []) else 0,
                "Difficulty Breathing": 1 if "Difficulty breathing" in u.get("symptoms", []) else 0,
                "bmi": u.get("bmi", 0),
                "systolic_bp": u.get("systolic_bp", 0),
                "diastolic_bp": u.get("diastolic_bp", 0),
                "heart_rate": u.get("resting_hr", 0),
                "smoker": 1 if u.get("smoking_status") == "Current smoker" else 0,
                "exercise_frequency": u.get("exercise_days", 0),
                "diabetes": 1 if "Diabetes" in u.get("chronic_conditions", []) else 0,
                "hypertension": 1 if "Hypertension" in u.get("chronic_conditions", []) else 0,
                "heart_disease": 1 if "Heart disease" in u.get("chronic_conditions", []) else 0,
                "has_fever": 1 if "Fever" in u.get("symptoms", []) else 0,
                "has_cough": 1 if "Cough" in u.get("symptoms", []) else 0,
                "has_fatigue": 1 if "Fatigue" in u.get("symptoms", []) else 0,
                "has_breathing_difficulty": 1 if "Difficulty breathing" in u.get("symptoms", []) else 0
            }
            df = pd.DataFrame([df_risk])
            try:
                risk = predict_cluster(df)
            except:
                risk = "unknown"
            all_risks.append(risk)
        risk_df = pd.DataFrame({"Risk Category": all_risks})
        st.plotly_chart(px.pie(risk_df, names="Risk Category", title="Users by Risk Category"))

        # Create tabs for different admin views
        tab1, tab2, tab3 = st.tabs(["Risk Categories", "User Details", "Reports"])
        
        with tab1:
            st.header("Patient Risk Categories")
            
            # Categorize users by risk level
            risk_categories = {
                "high": [],
                "moderate": [],
                "low": [],
                "very low": [],
                "very high": [],
                "high risk": [],
                "moderate risk": [],
                "low risk": [],
                "very low risk": []
            }
            
            for user in users:
                # Create DataFrame for clustering
                user_data = {
                    "Age": user.get('age', 0),
                    "Gender": user.get('gender', ''),
                    "Blood Pressure": user.get('bp_category', 'Normal'),
                    "Cholesterol Level": user.get('cholesterol', 'Normal'),
                    "Fever": 1 if "Fever" in user.get('symptoms', []) else 0,
                    "Cough": 1 if "Cough" in user.get('symptoms', []) else 0,
                    "Fatigue": 1 if "Fatigue" in user.get('symptoms', []) else 0,
                    "Difficulty Breathing": 1 if "Difficulty breathing" in user.get('symptoms', []) else 0,
                    "age": user.get('age', 0),
                    "gender": user.get('gender', ''),
                    "bmi": user.get('bmi', 0),
                    "systolic_bp": user.get('systolic_bp', 0),
                    "diastolic_bp": user.get('diastolic_bp', 0),
                    "heart_rate": user.get('resting_hr', 0),
                    "smoker": 1 if user.get('smoking_status') == "Current smoker" else 0,
                    "exercise_frequency": user.get('exercise_days', 0),
                    "diabetes": 1 if "Diabetes" in user.get('chronic_conditions', []) else 0,
                    "hypertension": 1 if "Hypertension" in user.get('chronic_conditions', []) else 0,
                    "heart_disease": 1 if "Heart disease" in user.get('chronic_conditions', []) else 0,
                    "has_fever": 1 if "Fever" in user.get('symptoms', []) else 0,
                    "has_cough": 1 if "Cough" in user.get('symptoms', []) else 0,
                    "has_fatigue": 1 if "Fatigue" in user.get('symptoms', []) else 0,
                    "has_breathing_difficulty": 1 if "Difficulty breathing" in user.get('symptoms', []) else 0
                }
                df_row = pd.DataFrame([user_data])
                risk_level = predict_cluster(df_row)
                risk_categories[risk_level].append(user)
            
            # Display each risk category
            for risk_level, patients in risk_categories.items():
                with st.expander(f"{risk_level.upper()} RISK ({len(patients)} patients)"):
                    if patients:
                        for patient in patients:
                            st.write(f"Name: {patient.get('name', 'N/A')}")
                            st.write(f"Email: {patient.get('email', 'N/A')}")
                            st.write("---")
                    else:
                        st.write("No patients in this category")
        
        with tab2:
            st.header("User Details")
            # Create a search box
            search_query = st.text_input("Search users by name or email")
            
            # Filter users based on search query
            filtered_users = users
            if search_query:
                filtered_users = [
                    user for user in users 
                    if search_query.lower() in user.get('name', '').lower() 
                    or search_query.lower() in user.get('email', '').lower()
                ]
            
            # Display user details in a table
            if filtered_users:
                user_data = []
                for user in users:
                    # Create DataFrame for clustering
                    user_data_for_clustering = {
                        "Age": user.get('age', 0),
                        "Gender": user.get('gender', ''),
                        "Blood Pressure": user.get('bp_category', 'Normal'),
                        "Cholesterol Level": user.get('cholesterol', 'Normal'),
                        "Fever": 1 if "Fever" in user.get('symptoms', []) else 0,
                        "Cough": 1 if "Cough" in user.get('symptoms', []) else 0,
                        "Fatigue": 1 if "Fatigue" in user.get('symptoms', []) else 0,
                        "Difficulty Breathing": 1 if "Difficulty breathing" in user.get('symptoms', []) else 0,
                        "age": user.get('age', 0),
                        "gender": user.get('gender', ''),
                        "bmi": user.get('bmi', 0),
                        "systolic_bp": user.get('systolic_bp', 0),
                        "diastolic_bp": user.get('diastolic_bp', 0),
                        "heart_rate": user.get('resting_hr', 0),
                        "smoker": 1 if user.get('smoking_status') == "Current smoker" else 0,
                        "exercise_frequency": user.get('exercise_days', 0),
                        "diabetes": 1 if "Diabetes" in user.get('chronic_conditions', []) else 0,
                        "hypertension": 1 if "Hypertension" in user.get('chronic_conditions', []) else 0,
                        "heart_disease": 1 if "Heart disease" in user.get('chronic_conditions', []) else 0,
                        "has_fever": 1 if "Fever" in user.get('symptoms', []) else 0,
                        "has_cough": 1 if "Cough" in user.get('symptoms', []) else 0,
                        "has_fatigue": 1 if "Fatigue" in user.get('symptoms', []) else 0,
                        "has_breathing_difficulty": 1 if "Difficulty breathing" in user.get('symptoms', []) else 0
                    }
                    df_row = pd.DataFrame([user_data_for_clustering])
                    risk_level = predict_cluster(df_row)
                    user_data.append({
                        'Name': user.get('name', 'N/A'),
                        'Email': user.get('email', 'N/A'),
                        'Risk Level': risk_level,
                        'Last Updated': user.get('last_updated', 'N/A')
                    })
                st.dataframe(pd.DataFrame(user_data))
            else:
                st.write("No users found")
        
        with tab3:
            st.header("User Reports")
            # --- Global Report Search ---
            st.subheader("🔍 Global Report Search")
            global_query = st.text_input("Ask a question about all users' reports:")

            if global_query:
                from sentence_transformers import SentenceTransformer
                from sklearn.metrics.pairwise import cosine_similarity

                model = SentenceTransformer("all-MiniLM-L6-v2")
                all_reports = list(db.reports.find())
                if all_reports:
                    query_embedding = model.encode(global_query)
                    scored = []
                    for rpt in all_reports:
                        report_text = rpt.get("report_text", "")
                        report_embedding = model.encode(report_text)
                        score = cosine_similarity([query_embedding], [report_embedding])[0][0]
                        scored.append((rpt, score))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    top_hits = scored[:5]

                    st.markdown("### Top Matching Reports:")
                    for rpt, score in top_hits:
                        user = None
                        try:
                            user = db.users.find_one({"_id": ObjectId(rpt["user_id"])})
                        except Exception:
                            pass
                        user_name = user.get("name", "Unknown") if user else "Unknown"
                        report_date = rpt.get("created_at", "").strftime("%Y-%m-%d %H:%M") if isinstance(rpt.get("created_at", ""), datetime) else "Unknown Date"
                        with st.expander(f"{user_name} – {report_date} (Score: {score:.2f})"):
                            st.markdown(rpt.get("report_text", "No content available."))

            # Select user to view report
            user_emails = [user.get('email', 'N/A') for user in users]
            selected_email = st.selectbox("Select user to view report", user_emails)
            
            if selected_email != 'N/A':
                selected_user = next((user for user in users if user.get('email') == selected_email), None)
                if selected_user:
                    st.subheader(f"Report for {selected_user.get('name', 'N/A')}")
                    st.write("Uploaded Reports:")
                    reports = list(db.reports.find({"user_id": selected_user.get("_id")}))
                    if reports:
                        for report in sorted(reports, key=lambda x: x.get("created_at", datetime.min), reverse=True):
                            report_date = report.get("created_at", "").strftime("%Y-%m-%d %H:%M") if isinstance(report.get("created_at", ""), datetime) else "Unknown Date"
                            with st.expander(f"📄 Report from {report_date}"):
                                st.markdown(report.get("report_text", "No content available."))
                    else:
                        st.write("No reports found for this user.")
                    
                    # Display health data
                    if 'health_data' in selected_user:
                        st.write("Health Data:")
                        health_data = selected_user['health_data']
                        for key, value in health_data.items():
                            st.write(f"{key}: {value}")
                    
                    # Display medical records
                    if 'medical_records' in selected_user:
                        st.write("\nMedical Records:")
                        st.write(selected_user['medical_records'])
    else:
        st.error("Could not connect to the database. Please check your MongoDB connection.")

def main():
    """Main application function."""
    # Initialize session state
    if 'user_account' not in st.session_state:
        st.session_state['user_account'] = None
    if 'admin_logged_in' not in st.session_state:
        st.session_state['admin_logged_in'] = False
    
    # Check if admin is logged in
    if st.session_state['admin_logged_in']:
        show_admin_panel()
        if st.sidebar.button("Logout", key="admin_logout_button"):
            st.session_state['admin_logged_in'] = False
            st.rerun()
    else:
        # Show admin login in sidebar only if user is not logged in
        if st.session_state['user_account'] is None:
            admin_login()
        # Check if user is logged in
        if st.session_state['user_account'] is None:
            st.header("Login or Create Account")
            login_tab, create_account_tab = st.tabs(["Login", "Create Account"])
            
            with login_tab:
                st.subheader("Login")
                username = st.text_input("Email", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                if st.button("Login", key="user_login_button"):
                    login_user(username, password)
            
            with create_account_tab:
                st.subheader("Create Account")
                name = st.text_input("Name", key="create_name")
                email = st.text_input("Email", key="create_email")
                age = st.number_input("Age", min_value=0, max_value=120, value=30, key="create_age")
                gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"], key="create_gender")
                ethnicity = st.selectbox("Ethnicity", [
                    "White/Caucasian", 
                    "Black/African", 
                    "Hispanic/Latino", 
                    "East Asian", 
                    "South Asian",
                    "Middle Eastern", 
                    "Pacific Islander", 
                    "Native American", 
                    "Mixed", 
                    "Other", 
                    "Prefer not to say"
                ], key="create_ethnicity")
                weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0, step=0.1, key="create_weight")
                height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=170.0, step=0.5, key="create_height")
                bmi = calculate_bmi(weight, height)
                bmi_category = get_bmi_category(bmi)
                cholesterol = st.selectbox("Cholesterol Level", ["Normal", "High", "Unknown"], key="create_cholesterol")
                systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=0, max_value=250, value=120, key="create_systolic_bp")
                diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=0, max_value=150, value=80, key="create_diastolic_bp")
                bp_category = get_bp_category(systolic_bp, diastolic_bp)
                resting_hr = st.number_input("Resting Heart Rate (bpm)", min_value=0, max_value=200, value=70, key="create_resting_hr")
                smoking_status = st.selectbox("Smoking Status", [
                    "Never smoked", 
                    "Former smoker", 
                    "Current smoker"
                ], key="create_smoking_status")
                exercise_days = st.slider("Exercise Days/Week", 0, 7, 3, key="create_exercise_days")
                diet = st.text_area("Describe your diet", placeholder="e.g., vegetarian, low-carb, etc.", key="create_diet")
                symptoms = st.multiselect(
                    "Current Symptoms", 
                    ["None", "Fever", "Cough", "Fatigue", "Difficulty breathing", "Headache", "Dizziness", "Nausea", "Pain"],
                    key="create_symptoms"
                )
                chronic_conditions = st.multiselect(
                    "Chronic Conditions",
                    ["None", "Diabetes", "Hypertension", "Heart disease", "Asthma", "Cancer", "Arthritis", "Thyroid disorder"],
                    key="create_chronic_conditions"
                )
                password = st.text_input("Password", type="password", key="create_password")
                if st.button("Create Account", key="create_account_button"):
                    db = connect_to_mongodb()
                    if db is not None and db.users.find_one({"email": email}):
                        st.warning("An account with this email already exists.")
                    else:
                        create_user_account(name, email, age, gender, ethnicity, weight, height, bmi, bmi_category,
                                            cholesterol, systolic_bp, diastolic_bp, bp_category, resting_hr,
                                            smoking_status, exercise_days, diet, symptoms, chronic_conditions, password)
                        st.session_state['user_account'] = {
                            "name": name,
                            "email": email,
                            "age": age,
                            "gender": gender,
                            "ethnicity": ethnicity,
                            "weight_kg": weight,
                            "height_cm": height,
                            "bmi": bmi,
                            "bmi_category": bmi_category,
                            "cholesterol": cholesterol,
                            "systolic_bp": systolic_bp,
                            "diastolic_bp": diastolic_bp,
                            "bp_category": bp_category,
                            "resting_hr": resting_hr,
                            "smoking_status": smoking_status,
                            "exercise_days": exercise_days,
                            "diet": diet,
                            "symptoms": symptoms if "None" not in symptoms else [],
                            "chronic_conditions": chronic_conditions if "None" not in chronic_conditions else [],
                            "password": password
                        }
                        st.rerun()
        else:
            # Main content area
            st.header("Welcome, " + st.session_state['user_account']['name'] + "!")
            # User logout button
            st.sidebar.subheader("Account Settings")
            if st.sidebar.button("Logout", key="user_logout_button"):
                for key in list(st.session_state.keys()):
                    if key not in ['admin_logged_in']:
                        del st.session_state[key]
                st.success("Logged out successfully.")
                st.rerun()
            
            # PDF Upload
            st.subheader("Upload Medical Records")
            uploaded_files = st.file_uploader("Upload medical PDFs", type="pdf", accept_multiple_files=True, key="pdf_uploader")
            if uploaded_files:
                pdf_text = ""
                for uploaded_file in uploaded_files:
                    try:
                        # Create a PDF reader object
                        pdf_reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
                        
                        # Extract text from each page
                        for page in pdf_reader.pages:
                            pdf_text += page.extract_text() + "\n"
                            
                        st.success(f"Successfully processed {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                if pdf_text.strip():
                    st.session_state['medical_records'] = pdf_text
                    st.success("PDF files processed successfully!")
                else:
                    st.warning("No text could be extracted from the PDF files.")
            
            # Display uploaded PDFs
            st.subheader("Uploaded Medical Records")
            if st.session_state.get('medical_records'):
                st.text_area("Medical Records", st.session_state['medical_records'], height=200)
            else:
                st.info("No medical records uploaded yet.")
            
            # Create sidebar for user input
            with st.sidebar:
                st.header("Your Health Profile")
                
                # Basic Information
                st.subheader("Personal Information")
                name = st.text_input("Name", key="main_name")
                age = st.number_input("Age", min_value=0, max_value=120, value=30, key="main_age")
                gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"], key="main_gender")
                ethnicity = st.selectbox("Ethnicity", [
                    "White/Caucasian", 
                    "Black/African", 
                    "Hispanic/Latino", 
                    "East Asian", 
                    "South Asian",
                    "Middle Eastern", 
                    "Pacific Islander", 
                    "Native American", 
                    "Mixed", 
                    "Other", 
                    "Prefer not to say"
                ], key="main_ethnicity")
                
                # Body Measurements
                st.subheader("Body Measurements")
                weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0, step=0.1, key="main_weight")
                height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=170.0, step=0.5, key="main_height")
                
                # Calculate BMI
                bmi = calculate_bmi(weight, height)
                bmi_category = get_bmi_category(bmi)
                
                # Cholesterol level
                cholesterol = st.selectbox("Cholesterol Level", ["Normal", "High", "Unknown"], key="main_cholesterol")
                
                # Vitals
                st.subheader("Vitals")
                col1, col2 = st.columns(2)
                with col1:
                    systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=0, max_value=250, value=120, key="main_systolic_bp")
                with col2:
                    diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=0, max_value=150, value=80, key="main_diastolic_bp")
                
                bp_category = get_bp_category(systolic_bp, diastolic_bp)
                csv_bp_category = map_bp_to_csv_category(bp_category)
                resting_hr = st.number_input("Resting Heart Rate (bpm)", min_value=0, max_value=200, value=70, key="main_resting_hr")
                
                # Lifestyle
                st.subheader("Lifestyle")
                smoking_status = st.selectbox("Smoking Status", [
                    "Never smoked", 
                    "Former smoker", 
                    "Current smoker"
                ], key="main_smoking_status")
                exercise_days = st.slider("Exercise Days/Week", 0, 7, 3, key="main_exercise_days")
                diet = st.text_area("Describe your diet", placeholder="e.g., vegetarian, low-carb, etc.", key="main_diet")
                
                # Symptoms and Conditions
                st.subheader("Health Status")
                symptoms = st.multiselect(
                    "Current Symptoms", 
                    ["None", "Fever", "Cough", "Fatigue", "Difficulty breathing", "Headache", "Dizziness", "Nausea", "Pain"],
                    key="main_symptoms"
                )
                
                chronic_conditions = st.multiselect(
                    "Chronic Conditions",
                    ["None", "Diabetes", "Hypertension", "Heart disease", "Asthma", "Cancer", "Arthritis", "Thyroid disorder"],
                    key="main_chronic_conditions"
                )
                
                # Save to Database Button
                if st.button("Save to Database", key="save_to_db_button"):
                    user_data = st.session_state.get('user_account', {})
                    db = connect_to_mongodb()
                    if db is not None and not db.users.find_one({"email": user_data.get("email")}):
                        save_user_data_to_db(user_data)
                    else:
                        st.warning("User data already exists in the database.")

            # Main content area
            if st.button("Analyze Health Data", key="analyze_health_button"):
                if not name:
                    st.warning("Please enter your name to continue.")
                    return
                
                # Create progress indicator
                progress = st.progress(0)
                
                # Step 1: Extract PDF text
                progress.progress(20)
                st.subheader("Processing medical records...")
                pdf_text = st.session_state.get('medical_records', "")
                
                # Step 2: Prepare user data for clustering
                progress.progress(40)
                st.subheader("Building health profile...")
                
                # Store user data in a dictionary for easy access
                user_data = {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "ethnicity": ethnicity,
                    "weight_kg": weight,
                    "height_cm": height,
                    "bmi": bmi,
                    "bmi_category": bmi_category,
                    "cholesterol": cholesterol,
                    "systolic_bp": systolic_bp,
                    "diastolic_bp": diastolic_bp,
                    "bp_category": bp_category,
                    "resting_hr": resting_hr,
                    "smoking_status": smoking_status,
                    "exercise_days": exercise_days,
                    "diet": diet,
                    "symptoms": symptoms if "None" not in symptoms else [],
                    "chronic_conditions": chronic_conditions if "None" not in chronic_conditions else []
                }
                
                # Create DataFrame for clustering - matching dataset column names
                df_row = pd.DataFrame([{
                    "Age": age,
                    "Gender": gender,
                    "Blood Pressure": csv_bp_category,
                    "Cholesterol Level": cholesterol,
                    "Fever": 1 if "Fever" in symptoms else 0,
                    "Cough": 1 if "Cough" in symptoms else 0,
                    "Fatigue": 1 if "Fatigue" in symptoms else 0,
                    "Difficulty Breathing": 1 if "Difficulty breathing" in symptoms else 0,
                    # Include original columns for backwards compatibility
                    "age": age,
                    "gender": gender, 
                    "bmi": bmi,
                    "systolic_bp": systolic_bp,
                    "diastolic_bp": diastolic_bp,
                    "heart_rate": resting_hr,
                    "smoker": 1 if smoking_status == "Current smoker" else 0,
                    "exercise_frequency": exercise_days,
                    "diabetes": 1 if "Diabetes" in chronic_conditions else 0,
                    "hypertension": 1 if "Hypertension" in chronic_conditions else 0,
                    "heart_disease": 1 if "Heart disease" in chronic_conditions else 0,
                    "has_fever": 1 if "Fever" in symptoms else 0,
                    "has_cough": 1 if "Cough" in symptoms else 0,
                    "has_fatigue": 1 if "Fatigue" in symptoms else 0,
                    "has_breathing_difficulty": 1 if "Difficulty breathing" in symptoms else 0
                }])
                
                # Step 3: Predict cluster
                progress.progress(60)
                st.subheader("Analyzing risk factors...")
                cluster_label = predict_cluster(df_row)
                risk_factors = identify_risk_factors(user_data)
                
                # Step 4: Get recommendations from Gemini
                progress.progress(80)
                st.subheader("Generating recommendations...")
                recommendations = chat_with_gemini_with_context(
                    f"Based on the following health data, provide personalized health recommendations:\n\n{json.dumps(user_data, indent=2)}\n\nRisk Level: {cluster_label}\nRisk Factors: {', '.join(risk_factors)}\n\nMedical Records:\n{pdf_text}",
                    pdf_text
                )
                
                # Display results
                progress.progress(100)
                
                st.header("Your Health Assessment")
                
                # Display risk cluster with appropriate styling
                col1, col2, col3 = st.columns(3)
                with col1:
                    if cluster_label == "low":
                        st.success("Risk Level: LOW")
                    elif cluster_label == "moderate":
                        st.warning("Risk Level: MODERATE")
                    else:
                        st.error("Risk Level: HIGH")
                
                with col2:
                    st.metric("BMI", f"{bmi} ({bmi_category})")
                
                with col3:
                    st.metric("Blood Pressure", f"{systolic_bp}/{diastolic_bp} ({bp_category})")
                
                # Display risk factors
                if risk_factors:
                    st.subheader("Identified Risk Factors")
                    st.write(", ".join(risk_factors))
                
                # Display LLM recommendations
                st.subheader("Personalized Health Recommendations")
                st.markdown(recommendations)

                # Save the current report to MongoDB
                db = connect_to_mongodb()
                if db is not None:
                    report = {
                        "user_id": str(st.session_state['user_account'].get('_id', '')),
                        "report_text": recommendations,
                        "risk_cluster": cluster_label,
                        "risk_factors": risk_factors,
                        "pdf_text": pdf_text,
                        "created_at": datetime.utcnow()
                    }
                    db.reports.insert_one(report)
                    st.session_state['report_saved'] = True
                    st.success("Report saved successfully!")


            # --- Query Past Reports Using RAG ---
            st.subheader("📄 Your Past Reports")
            db = connect_to_mongodb()
            user_reports = []
            if db is not None:
                user_reports = list(db.reports.find({"user_id": str(st.session_state['user_account'].get('_id', ''))}))
            if user_reports:
                st.subheader("📋 Click to View Individual Reports")
                for i, rpt in enumerate(sorted(user_reports, key=lambda x: x.get("created_at", datetime.min), reverse=True)):
                    date_str = rpt.get("created_at", "").strftime("%Y-%m-%d %H:%M") if isinstance(rpt.get("created_at", ""), datetime) else "Unknown Date"
                    with st.expander(f"📝 Report from {date_str}"):
                        st.markdown(rpt.get('report_text', 'No content available.'))

                        # Download button
                        download_text = rpt.get('report_text', '')
                        st.download_button(
                            label="📥 Download Report",
                            data=download_text,
                            file_name=f"health_report_{date_str.replace(' ', '_').replace(':', '-')}.txt",
                            mime="text/plain",
                            key=f"download_{i}"
                        )

                        # Delete button
                        if st.button(f"🗑️ Delete This Report", key=f"delete_{i}"):
                            db.reports.delete_one({"_id": rpt["_id"]})
                            st.success("Report deleted.")
                            st.rerun()
            else:
                st.info("No previous reports found.")

            # --- Query Past Reports Using RAG ---
            st.subheader("💬 Ask About Your Past Reports")
            user_question = st.text_input("Enter your question about previous health reports:", key="report_query_input")
            if st.button("Ask", key="report_query_button") and user_question:
                from sentence_transformers import SentenceTransformer
                from sklearn.metrics.pairwise import cosine_similarity

                model = SentenceTransformer("all-MiniLM-L6-v2")
                user_reports = list(db.reports.find({"user_id": str(st.session_state['user_account'].get('_id', ''))}))

                if user_reports:
                    query_embedding = model.encode(user_question)

                    # Compute similarity with each report
                    scored = []
                    for rpt in user_reports:
                        report_embedding = model.encode(rpt["report_text"])
                        score = cosine_similarity([query_embedding], [report_embedding])[0][0]
                        scored.append((rpt, score))
                    scored.sort(key=lambda x: x[1], reverse=True)

                    # Use top 3 reports as context
                    top_reports = [r[0]["report_text"] for r in scored[:3]]
                    joined_reports = "\n\n---\n\n".join(top_reports)
                    prompt = f"""
You are a concise and factual health assistant. Your goal is to answer the user's question *only* using the provided reports. 
If the reports do not contain enough information, say "This information is not available in your report."

--- Begin Reports ---
{joined_reports}
--- End Reports ---

User Question: {user_question}
Answer:
"""
                    response = chat_with_gemini(prompt)
                    st.markdown(f"**Health Assistant:** {response}")
                else:   
                    st.info("No previous reports found to query.")

            # --- Minimal Chat UI ---
            st.markdown("---")
            st.header("Ask a Health Question (Chat)")
            if 'chat_history' not in st.session_state:
                st.session_state['chat_history'] = []
            user_chat_input = st.text_input("Type your question here:", key="chat_input")
            if st.button("Send", key="chat_send_button") and user_chat_input:
                response = chat_with_gemini_with_context(user_chat_input, st.session_state.get('medical_records', ""))
                st.session_state['chat_history'].append((user_chat_input, response))
            # Display chat history
            for i, (q, a) in enumerate(st.session_state['chat_history']):
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Health Assistant:** {a}")

if __name__ == "__main__":
    main() 