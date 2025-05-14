# Health Risk Assessment App

A Python application that analyzes personal health data, medical records, and provides personalized recommendations using machine learning and Gemini AI.

## Features

- Collects detailed health information through a Streamlit UI
- Analyzes PDF medical records
- Uses K-Means clustering to assess health risk levels
- Provides personalized recommendations via Gemini API

## Setup

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your Gemini API key:
   ```
   GOOGLE_API_KEY="YOUR API KEY"
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

## Dataset

The application automatically downloads the required disease symptoms dataset on first run. 
