"""Module for interfacing with the Gemini API."""
import os
import json
from typing import Dict, List, Any
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure the Gemini API
if API_KEY:
    genai.configure(api_key=API_KEY)

def get_recommendations(
    user_json: Dict[str, Any],
    cluster: str,
    risks: List[str],
    pdf_text: str
) -> str:
    """Get health recommendations from Gemini based on user profile and risk assessment.
    
    Args:
        user_json: Dictionary with user health data
        cluster: Risk level cluster ("low", "moderate", "high")
        risks: List of identified risk factors
        pdf_text: Extracted text from medical PDFs, or empty string
        
    Returns:
        Formatted string with Gemini's recommendations
    """
    if not API_KEY:
        return "Error: GOOGLE_API_KEY not found in .env file"
    
    # Create user message including all data and instructions
    user_message = f"""
You are a compassionate and knowledgeable healthcare assistant with expertise in internal medicine, lifestyle coaching, and interpreting lab reports.

Your task is to analyze the user's profile and provide detailed, structured, and medically sound insights across three sections.

Respond using **clear bullet points**, plain English, and **short paragraphs** when appropriate.

---

**Section 1 – Overview**
- Summarize the user's health status and explain the assigned risk cluster.
- Include general health observations, pre-existing conditions (if any), and notable lifestyle aspects.
- Keep it concise but insightful, highlighting how the user's profile relates to their risk level.

---

**Section 2 – Personalized Action Plan**
Provide 5–8 bullet points, **ranked from easiest to implement to most medically urgent**:
- Start with simple lifestyle changes (e.g. hydration, walking, diet tweaks).
- Progress to moderate actions (e.g. structured exercise, sleep hygiene, diet overhaul).
- End with critical interventions (e.g. doctor visits, medications, screenings).
- Where helpful, add estimated timelines, frequencies (e.g., "walk 30 minutes 5x/week"), or simple metrics to track.

---

**Section 3 – Lab Report Interpretation**
- If lab/imaging data is present in the input text, extract and interpret it clearly.
- Flag any **abnormal values** and **explain what they mean** in plain terms (e.g., "Your hemoglobin is low, which may suggest anemia").
- Clarify medical jargon for a non-technical reader.
- If no lab data is provided, return: **"(no lab records uploaded)"**
    
    USER PROFILE:
    {json.dumps(user_json, indent=2)}
    
    RISK CLUSTER: {cluster}
    
    IDENTIFIED RISK FACTORS:
    {', '.join(risks)}
    
    MEDICAL RECORDS TEXT:
    {pdf_text if pdf_text else "(no medical records uploaded)"}
    """
    
    # Call Gemini API
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": 0.4,
            "max_output_tokens": 8192,
        }
    )
    
    try:
        response = model.generate_content(user_message)
        
        # Return the text response
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"

def chat_with_gemini(user_message: str) -> str:
    """Send a general chat message to Gemini and return the response, always as an expert doctor, and only answer medical queries."""
    if not API_KEY:
        return "Error: GOOGLE_API_KEY not found in .env file"
    system_prompt = (
        "You are a highly experienced, compassionate medical doctor. "
        "Only answer questions that are related to health, medicine, symptoms, treatments, or medical science. "
        "If the user asks a non-medical question, politely respond: 'I'm here to help with medical and health-related questions only.' "
        "Always provide clear, evidence-based, and responsible medical information. "
        "If a question requires a diagnosis or treatment, remind the user to consult a healthcare professional for personal medical advice."
    )
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 512,
        }
    )
    try:
        full_prompt = f"{system_prompt}\n\nUser: {user_message}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"

def chat_with_gemini_with_context(user_input, medical_records):
    """Chat with Gemini using the provided medical records as context."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    chat = model.start_chat(history=[])
    system_prompt = (
        "You are a highly experienced, compassionate medical doctor. "
        "Only answer questions that are related to health, medicine, symptoms, treatments, or medical science. "
        "If the user asks a non-medical question, politely respond: 'I'm here to help with medical and health-related questions only.' "
        "Always provide clear, evidence-based, and responsible medical information. "
        "If a question requires a diagnosis or treatment, remind the user to consult a healthcare professional for personal medical advice. "
        "IMPORTANT: You MUST analyze any provided medical records and lab reports in detail. "
        "For each medical record or lab report provided, you should: "
        "1. Identify and explain key findings and values "
        "2. Compare values to normal ranges "
        "3. Highlight any concerning or abnormal results "
        "4. Provide context for the results "
        "5. Suggest follow-up actions if needed "
        "Always include this analysis in your response when medical records are provided."
    )
    
    # Format the context to make it clear to the model
    formatted_context = f"""
MEDICAL RECORDS AND LAB REPORTS:
--------------------------------
{medical_records}
--------------------------------

USER QUESTION:
{user_input}
"""
    
    response = chat.send_message(
        f"{system_prompt}\n\n{formatted_context}",
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=2048,
            temperature=0.7,
            top_p=0.8,
            top_k=40
        )
    )
    return response.text

def save_user_data_to_db(user_data):
    """Save user data to MongoDB."""
    db = connect_to_mongodb()
    db.users.update_one({"_id": user_data.get("_id")}, {"$set": user_data}, upsert=True)
    st.success("Data saved to database successfully!")

def create_user_account(name, age, gender, ethnicity, weight, height, bmi, bmi_category, cholesterol, systolic_bp, diastolic_bp, bp_category, resting_hr, smoking_status, exercise_days, diet, symptoms, chronic_conditions, password):
    """Simulate creating a user account by storing user data in session state."""
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
        "chronic_conditions": chronic_conditions if "None" not in chronic_conditions else [],
        "password": password
    }
    db = connect_to_mongodb()
    result = db.users.insert_one(user_data)
    user_data["_id"] = str(result.inserted_id)
    st.session_state['user_account'] = user_data
    st.session_state['name'] = name  # Store the name in session state
    st.success("Account created successfully!")

def main():
    """Main application function."""
    st.title("Health Risk Assessment & Personalized Recommendations")
    
    # Debug: Print session state
    st.write("Session State:", st.session_state)
    
    # Check if user is logged in
    if 'user_account' not in st.session_state:
        st.header("Login or Create Account")
        login_tab, create_account_tab = st.tabs(["Login", "Create Account"])
        
        with login_tab:
            st.subheader("Login")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                login_user(username, password)
        
        with create_account_tab:
            st.subheader("Create Account")
            name = st.text_input("Name", key="create_name")
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
            if st.button("Create Account"):
                create_user_account(name, age, gender, ethnicity, weight, height, bmi, bmi_category, cholesterol, systolic_bp, diastolic_bp, bp_category, resting_hr, smoking_status, exercise_days, diet, symptoms, chronic_conditions, password)
    else:
        # Main content area
        st.header("Welcome, " + st.session_state['user_account']['name'] + "!")
        
        # Display uploaded PDFs
        st.subheader("Uploaded Medical Records")
        if st.session_state.get('medical_records'):
            st.text_area("Medical Records", st.session_state['medical_records'], height=200)
        else:
            st.info("No medical records uploaded yet.")
        
        # Display user data from database
        st.subheader("Your Health Data")
        user_data = st.session_state.get('user_account', {})
        if user_data:
            st.json(user_data)
        else:
            st.info("No user data available.")
        
        # Create sidebar for user input
        with st.sidebar:
            st.header("Your Health Profile")
            
            # Basic Information
            st.subheader("Personal Information")
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
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
            ])
            
            # Body Measurements
            st.subheader("Body Measurements")
            weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0, step=0.1)
            height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=170.0, step=0.5)
            
            # Calculate BMI
            bmi = calculate_bmi(weight, height)
            bmi_category = get_bmi_category(bmi)
            
            # Cholesterol level
            cholesterol = st.selectbox("Cholesterol Level", ["Normal", "High", "Unknown"])
            
            # Vitals
            st.subheader("Vitals")
            col1, col2 = st.columns(2)
            with col1:
                systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=0, max_value=250, value=120)
            with col2:
                diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=0, max_value=150, value=80)
            
            bp_category = get_bp_category(systolic_bp, diastolic_bp)
            csv_bp_category = map_bp_to_csv_category(bp_category)
            resting_hr = st.number_input("Resting Heart Rate (bpm)", min_value=0, max_value=200, value=70)
            
            # Lifestyle
            st.subheader("Lifestyle")
            smoking_status = st.selectbox("Smoking Status", [
                "Never smoked", 
                "Former smoker", 
                "Current smoker"
            ])
            exercise_days = st.slider("Exercise Days/Week", 0, 7, 3)
            diet = st.text_area("Describe your diet", placeholder="e.g., vegetarian, low-carb, etc.")
            
            # Symptoms and Conditions
            st.subheader("Health Status")
            symptoms = st.multiselect(
                "Current Symptoms", 
                ["None", "Fever", "Cough", "Fatigue", "Difficulty breathing", "Headache", "Dizziness", "Nausea", "Pain"]
            )
            
            # Map to dataset symptom format
            has_fever = "Fever" in symptoms
            has_cough = "Cough" in symptoms
            has_fatigue = "Fatigue" in symptoms
            has_breathing_difficulty = "Difficulty breathing" in symptoms
            
            chronic_conditions = st.multiselect(
                "Chronic Conditions",
                ["None", "Diabetes", "Hypertension", "Heart disease", "Asthma", "Cancer", "Arthritis", "Thyroid disorder"]
            )
            
            other_condition = st.text_input("Other health conditions", "")
            if other_condition and "None" not in chronic_conditions:
                chronic_conditions.append(other_condition)
            
            # PDF Upload
            st.subheader("Medical Records")
            uploaded_files = st.file_uploader("Upload medical PDFs", type="pdf", accept_multiple_files=True)
            if uploaded_files:
                pdf_text = pdf_utils.extract_text(uploaded_files)
                st.session_state['medical_records'] = pdf_text
            else:
                st.session_state['medical_records'] = ""
            
            # Save to Database Button
            if st.button("Save to Database"):
                user_data = st.session_state.get('user_account', {})
                save_user_data_to_db(user_data)
            
            # View Database Button
            if st.button("View Database"):
                view_database()
        
        # Main content area
        if st.button("Analyze Health Data"):
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
                "Fever": 1 if has_fever else 0,
                "Cough": 1 if has_cough else 0,
                "Fatigue": 1 if has_fatigue else 0,
                "Difficulty Breathing": 1 if has_breathing_difficulty else 0,
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
                "has_fever": 1 if has_fever else 0,
                "has_cough": 1 if has_cough else 0,
                "has_fatigue": 1 if has_fatigue else 0,
                "has_breathing_difficulty": 1 if has_breathing_difficulty else 0
            }])
            
            # Step 3: Predict cluster
            progress.progress(60)
            st.subheader("Analyzing risk factors...")
            cluster_label = clustering.predict_cluster(df_row)
            risk_factors = identify_risk_factors(user_data)
            
            # Step 4: Get recommendations from Gemini
            progress.progress(80)
            st.subheader("Generating recommendations...")
            recommendations = llm_utils.get_recommendations(
                user_data, 
                cluster_label, 
                risk_factors, 
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

            # Download Report Button
            if st.button("Download Report"):
                user_data = st.session_state.get('user_account', {})
                medical_records = st.session_state.get('medical_records', "")
                report = download_report(user_data, medical_records)
                st.write("Report:", report)  # Debug: Print the report
                st.download_button(
                    label="Download Report as .txt",
                    data=report,
                    file_name="health_report.txt",
                    mime="text/plain"
                )
        
        # --- Minimal Chat UI ---
        st.markdown("---")
        st.header("Ask a Health Question (Chat)")
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        user_chat_input = st.text_input("Type your question here:", key="chat_input")
        if st.button("Send", key="chat_send") and user_chat_input:
            from llm_utils import chat_with_gemini_with_context
            medical_records = st.session_state.get('medical_records', "")
            response = chat_with_gemini_with_context(user_chat_input, medical_records)
            st.session_state['chat_history'].append((user_chat_input, response))
        # Display chat history
        for i, (q, a) in enumerate(st.session_state['chat_history']):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Gemini:** {a}")

if __name__ == "__main__":
    main() 