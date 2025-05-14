"""Module for K-Means clustering on disease symptom data."""
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import urllib.request
from functools import lru_cache

# Dataset path
DATASET_PATH = "data/disease_symptoms.csv"

def ensure_dataset_exists():
    """Check if the dataset exists."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

@lru_cache(maxsize=1)
def get_clustering_model():
    """Load the dataset and create a fitted clustering model.
    
    Returns:
        Tuple of (clustering pipeline, mapping of cluster indices to risk levels)
    """
    ensure_dataset_exists()
    
    # Load the dataset
    df = pd.read_csv(DATASET_PATH)
    
    # Convert 'Yes'/'No' to 1/0 for symptom columns
    for col in ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']:
        if col in df.columns:
            # Handle both string and numeric values
            if df[col].dtype == 'object':
                df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Identify numeric and categorical features
    numeric_features = ['Age']
    for col in ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']:
        if col in df.columns:
            numeric_features.append(col)
            
    categorical_features = ['Gender']
    for col in ['Blood Pressure', 'Cholesterol Level']:
        if col in df.columns:
            categorical_features.append(col)
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Create clustering pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('cluster', KMeans(n_clusters=3, random_state=42))
    ])
    
    # Fit the pipeline
    pipeline.fit(df[numeric_features + categorical_features])
    
    # Determine risk level mapping based on outcome variable if present
    if 'Outcome Variable' in df.columns:
        # Get cluster predictions for each sample
        cluster_ids = pipeline.predict(df[numeric_features + categorical_features])
        
        # Calculate percentage of positive outcomes in each cluster
        df['cluster'] = cluster_ids
        positive_rates = df.groupby('cluster')['Outcome Variable'].apply(
            lambda x: (x == 'Positive').mean()
        ).sort_values()
        
        # Map clusters to risk levels based on positive rates
        risk_mapping = {
            positive_rates.index[0]: "low",
            positive_rates.index[1]: "moderate",
            positive_rates.index[2]: "high"
        }
    else:
        # If no outcome variable, use a simple severity score
        cluster_centers = pipeline.named_steps['cluster'].cluster_centers_
        severity_scores = np.sum(np.abs(cluster_centers), axis=1)
        sorted_indices = np.argsort(severity_scores)
        risk_mapping = {
            sorted_indices[0]: "low",
            sorted_indices[1]: "moderate", 
            sorted_indices[2]: "high"
        }
    
    return pipeline, risk_mapping, numeric_features + categorical_features

def predict_cluster(df_row):
    """Predict the risk cluster for a user based on their health data."""
    # Example logic for risk levels
    bmi = df_row['bmi'].iloc[0]
    systolic_bp = df_row['systolic_bp'].iloc[0]
    diastolic_bp = df_row['diastolic_bp'].iloc[0]
    heart_rate = df_row['heart_rate'].iloc[0]
    smoker = df_row['smoker'].iloc[0]
    exercise_frequency = df_row['exercise_frequency'].iloc[0]
    diabetes = df_row['diabetes'].iloc[0]
    hypertension = df_row['hypertension'].iloc[0]
    heart_disease = df_row['heart_disease'].iloc[0]
    has_fever = df_row['has_fever'].iloc[0]
    has_cough = df_row['has_cough'].iloc[0]
    has_fatigue = df_row['has_fatigue'].iloc[0]
    has_breathing_difficulty = df_row['has_breathing_difficulty'].iloc[0]

    # Calculate risk score
    risk_score = 0
    if bmi < 18.5 or bmi > 30:
        risk_score += 1
    if systolic_bp > 140 or diastolic_bp > 90:
        risk_score += 1
    if heart_rate > 100:
        risk_score += 1
    if smoker:
        risk_score += 1
    if exercise_frequency < 3:
        risk_score += 1
    if diabetes:
        risk_score += 1
    if hypertension:
        risk_score += 1
    if heart_disease:
        risk_score += 1
    if has_fever or has_cough or has_fatigue or has_breathing_difficulty:
        risk_score += 1

    # Determine risk level
    if risk_score == 0:
        return "very low"
    elif risk_score <= 2:
        return "low"
    elif risk_score <= 4:
        return "moderate"
    elif risk_score <= 6:
        return "high"
    else:
        return "very high" 