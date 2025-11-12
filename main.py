from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict

# --- MLOps METADATA ---
MODEL_METADATA = {
    "version": "1.0.0",
    "algorithm": "Random Forest Classifier",
    "training_date": "2024-03-15",
    "metrics": {
        "accuracy": 0.932,
        "f1_score": 0.880,
        "precision": 0.891,
        "recall": 0.869
    },
    "required_features": ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
}

# Feature ranges (for validation and risk assessment)
FEATURE_RANGES = {
    "Pregnancies": {"min": 0, "max": 17, "normal": (0, 10)},
    "Glucose": {"min": 0, "max": 200, "normal": (70, 100), "prediabetic": (100, 125)},
    "BloodPressure": {"min": 0, "max": 122, "normal": (60, 80), "elevated": (80, 90)},
    "BMI": {"min": 0, "max": 67.1, "normal": (18.5, 24.9), "overweight": (25, 29.9)},
    "Age": {"min": 21, "max": 81, "normal": (21, 65)}
}
# --- END MLOps METADATA ---

app = FastAPI(
    title="Diabetes Prediction API - Enhanced",
    description="Advanced MLOps API for diabetes prediction with visualization and analytics",
    version=MODEL_METADATA["version"]
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for predictions and monitoring
prediction_logs = []
api_stats = defaultdict(int)

# Load the trained ML model
try:
    model = joblib.load("diabetes_model.pkl")
except FileNotFoundError:
    print("WARNING: 'diabetes_model.pkl' not found. API endpoints will fail.")
    model = None

# ==================== PYDANTIC MODELS ====================

class DiabetesInput(BaseModel):
    """Input features for prediction."""
    Pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    Glucose: float = Field(..., ge=0, le=200, description="Plasma glucose concentration")
    BloodPressure: float = Field(..., ge=0, le=122, description="Diastolic blood pressure (mm Hg)")
    BMI: float = Field(..., ge=0, le=70, description="Body mass index")
    Age: int = Field(..., ge=21, le=120, description="Age in years")

class BatchDiabetesInput(BaseModel):
    """Batch prediction input."""
    patients: List[DiabetesInput]

class PredictionOutput(BaseModel):
    """Prediction response."""
    prediction: str
    probability: float
    risk_score: float
    confidence_level: str

class DetailedPredictionOutput(BaseModel):
    """Detailed prediction with explanations."""
    prediction: str
    probability: float
    risk_score: float
    confidence_level: str
    feature_contributions: Dict[str, float]
    risk_factors: List[str]
    recommendations: List[str]
    timestamp: str

class FeatureImportanceOutput(BaseModel):
    """Feature importance response."""
    feature: str
    importance: float

class RiskAssessmentOutput(BaseModel):
    """Risk assessment for each feature."""
    feature: str
    value: float
    status: str
    risk_level: str
    normal_range: str

class WhatIfOutput(BaseModel):
    """What-if analysis response."""
    original_prediction: str
    original_probability: float
    modified_prediction: str
    modified_probability: float
    probability_change: float

class HealthRecommendation(BaseModel):
    """Health recommendation response."""
    category: str
    recommendations: List[str]
    priority: str

# ==================== HELPER FUNCTIONS ====================

def calculate_risk_score(probability: float) -> float:
    """Convert probability to a 0-100 risk score."""
    return round(probability * 100, 2)

def get_confidence_level(probability: float) -> str:
    """Determine confidence level based on probability."""
    if probability < 0.3 or probability > 0.7:
        return "High"
    elif probability < 0.4 or probability > 0.6:
        return "Medium"
    else:
        return "Low"

def assess_feature_risk(feature: str, value: float) -> Dict[str, Any]:
    """Assess risk level for a specific feature."""
    ranges = FEATURE_RANGES[feature]
    
    if feature == "Glucose":
        if value < ranges["normal"][0]:
            return {"status": "Low", "risk_level": "Low", "normal_range": f"{ranges['normal'][0]}-{ranges['normal'][1]}"}
        elif value <= ranges["normal"][1]:
            return {"status": "Normal", "risk_level": "Low", "normal_range": f"{ranges['normal'][0]}-{ranges['normal'][1]}"}
        elif value <= ranges["prediabetic"][1]:
            return {"status": "Prediabetic", "risk_level": "Medium", "normal_range": f"{ranges['normal'][0]}-{ranges['normal'][1]}"}
        else:
            return {"status": "High", "risk_level": "High", "normal_range": f"{ranges['normal'][0]}-{ranges['normal'][1]}"}
    
    elif feature == "BloodPressure":
        if value <= ranges["normal"][1]:
            return {"status": "Normal", "risk_level": "Low", "normal_range": f"{ranges['normal'][0]}-{ranges['normal'][1]}"}
        elif value <= ranges["elevated"][1]:
            return {"status": "Elevated", "risk_level": "Medium", "normal_range": f"{ranges['normal'][0]}-{ranges['normal'][1]}"}
        else:
            return {"status": "High", "risk_level": "High", "normal_range": f"{ranges['normal'][0]}-{ranges['normal'][1]}"}
    
    elif feature == "BMI":
        if value < ranges["normal"][0]:
            return {"status": "Underweight", "risk_level": "Low", "normal_range": f"{ranges['normal'][0]}-{ranges['normal'][1]}"}
        elif value <= ranges["normal"][1]:
            return {"status": "Normal", "risk_level": "Low", "normal_range": f"{ranges['normal'][0]}-{ranges['normal'][1]}"}
        elif value <= ranges["overweight"][1]:
            return {"status": "Overweight", "risk_level": "Medium", "normal_range": f"{ranges['normal'][0]}-{ranges['normal'][1]}"}
        else:
            return {"status": "Obese", "risk_level": "High", "normal_range": f"{ranges['normal'][0]}-{ranges['normal'][1]}"}
    
    else:
        if ranges["normal"][0] <= value <= ranges["normal"][1]:
            return {"status": "Normal", "risk_level": "Low", "normal_range": f"{ranges['normal'][0]}-{ranges['normal'][1]}"}
        else:
            return {"status": "Outside Normal", "risk_level": "Medium", "normal_range": f"{ranges['normal'][0]}-{ranges['normal'][1]}"}

def generate_recommendations(data: DiabetesInput, prediction: str, risk_factors: List[str]) -> List[HealthRecommendation]:
    """Generate personalized health recommendations."""
    recommendations = []
    
    # Glucose recommendations
    if data.Glucose > 125:
        recommendations.append({
            "category": "Blood Sugar Management",
            "recommendations": [
                "Monitor blood glucose levels regularly",
                "Reduce intake of refined carbohydrates and sugary foods",
                "Consider consulting an endocrinologist",
                "Increase fiber intake through whole grains and vegetables"
            ],
            "priority": "High"
        })
    elif data.Glucose > 100:
        recommendations.append({
            "category": "Blood Sugar Management",
            "recommendations": [
                "Limit sugar and refined carbohydrate intake",
                "Regular blood glucose monitoring recommended",
                "Maintain a balanced diet with complex carbohydrates"
            ],
            "priority": "Medium"
        })
    
    # BMI recommendations
    if data.BMI > 30:
        recommendations.append({
            "category": "Weight Management",
            "recommendations": [
                "Consult a nutritionist for a personalized meal plan",
                "Aim for gradual weight loss (1-2 lbs per week)",
                "Incorporate 150 minutes of moderate exercise weekly",
                "Consider joining a weight management program"
            ],
            "priority": "High"
        })
    elif data.BMI > 25:
        recommendations.append({
            "category": "Weight Management",
            "recommendations": [
                "Increase physical activity to 30 minutes daily",
                "Focus on portion control and balanced meals",
                "Track calorie intake using a food diary"
            ],
            "priority": "Medium"
        })
    
    # Blood pressure recommendations
    if data.BloodPressure > 90:
        recommendations.append({
            "category": "Blood Pressure",
            "recommendations": [
                "Reduce sodium intake to less than 2,300mg daily",
                "Monitor blood pressure regularly at home",
                "Consult with a physician about hypertension management",
                "Practice stress reduction techniques"
            ],
            "priority": "High"
        })
    elif data.BloodPressure > 80:
        recommendations.append({
            "category": "Blood Pressure",
            "recommendations": [
                "Reduce sodium intake in diet",
                "Monitor blood pressure monthly",
                "Increase potassium-rich foods"
            ],
            "priority": "Medium"
        })
    
    # General recommendations
    recommendations.append({
        "category": "General Health",
        "recommendations": [
            "Schedule regular check-ups with healthcare provider",
            "Stay hydrated with at least 8 glasses of water daily",
            "Get 7-9 hours of quality sleep per night",
            "Manage stress through meditation or yoga",
            "Avoid smoking and limit alcohol consumption"
        ],
        "priority": "Medium"
    })
    
    if prediction == "Diabetic":
        recommendations.append({
            "category": "Diabetes Management",
            "recommendations": [
                "Consult an endocrinologist immediately",
                "Learn about diabetes self-management education",
                "Consider continuous glucose monitoring",
                "Develop a comprehensive diabetes care plan"
            ],
            "priority": "High"
        })
    
    return recommendations

def calculate_feature_contributions(data: DiabetesInput, probability: float) -> Dict[str, float]:
    """Calculate approximate feature contributions to the prediction."""
    if model is None:
        return {}
    
    importances = model.feature_importances_
    features = MODEL_METADATA["required_features"]
    input_values = [data.Pregnancies, data.Glucose, data.BloodPressure, data.BMI, data.Age]
    
    # Normalize input values
    normalized_values = []
    for feature, value in zip(features, input_values):
        range_info = FEATURE_RANGES[feature]
        normalized = (value - range_info["min"]) / (range_info["max"] - range_info["min"])
        normalized_values.append(normalized)
    
    # Calculate contributions (simplified approach)
    contributions = {}
    total = sum(imp * norm * probability for imp, norm in zip(importances, normalized_values))
    
    for feature, importance, norm_val in zip(features, importances, normalized_values):
        contribution = (importance * norm_val * probability) / total if total > 0 else 0
        contributions[feature] = round(contribution * 100, 2)
    
    return contributions

# ==================== API ENDPOINTS ====================

# --- 1. HEALTH CHECK ENDPOINT ---
@app.get("/health", summary="Health check endpoint")
def health_check():
    """Returns the status of the API and model loading."""
    api_stats["health_check"] += 1
    status = "OK" if model is not None else "ERROR: Model not loaded"
    return {
        "status": status, 
        "service": "diabetes-api-enhanced",
        "timestamp": datetime.now().isoformat()
    }

# --- 2. MODEL INFO ENDPOINT ---
@app.get("/model_info", response_model=Dict[str, Any], summary="Get model metadata")
def get_model_info():
    """Returns detailed metadata about the deployed model."""
    api_stats["model_info"] += 1
    return MODEL_METADATA

# --- 3. BASIC PREDICT ENDPOINT ---
@app.post("/predict", response_model=PredictionOutput, summary="Basic diabetes prediction")
def predict(data: DiabetesInput):
    """Basic prediction endpoint with probability and risk score."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    api_stats["predict"] += 1
    
    input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.BMI, data.Age]])
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    probability_diabetic = probabilities[1]
    
    prediction_result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    risk_score = calculate_risk_score(probability_diabetic)
    confidence = get_confidence_level(probability_diabetic)
    
    # Log prediction
    prediction_logs.append({
        "timestamp": datetime.now().isoformat(),
        "input": data.dict(),
        "prediction": prediction_result,
        "probability": probability_diabetic
    })
    
    return {
        "prediction": prediction_result,
        "probability": round(probability_diabetic, 4),
        "risk_score": risk_score,
        "confidence_level": confidence
    }

# --- 4. DETAILED PREDICT ENDPOINT ---
@app.post("/predict/detailed", response_model=DetailedPredictionOutput, summary="Detailed prediction with explanations")
def predict_detailed(data: DiabetesInput):
    """Detailed prediction with feature contributions, risk factors, and recommendations."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    api_stats["predict_detailed"] += 1
    
    input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.BMI, data.Age]])
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    probability_diabetic = probabilities[1]
    
    prediction_result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    risk_score = calculate_risk_score(probability_diabetic)
    confidence = get_confidence_level(probability_diabetic)
    
    # Calculate feature contributions
    contributions = calculate_feature_contributions(data, probability_diabetic)
    
    # Identify risk factors
    risk_factors = []
    if data.Glucose > 125:
        risk_factors.append("High glucose levels detected")
    elif data.Glucose > 100:
        risk_factors.append("Elevated glucose levels (prediabetic range)")
    
    if data.BMI > 30:
        risk_factors.append("BMI indicates obesity")
    elif data.BMI > 25:
        risk_factors.append("BMI indicates overweight")
    
    if data.BloodPressure > 90:
        risk_factors.append("High blood pressure detected")
    elif data.BloodPressure > 80:
        risk_factors.append("Elevated blood pressure")
    
    if data.Age > 45:
        risk_factors.append("Age is a risk factor for diabetes")
    
    # Generate recommendations
    recommendations_list = generate_recommendations(data, prediction_result, risk_factors)
    flat_recommendations = []
    for rec in recommendations_list:
        for r in rec["recommendations"]:
            flat_recommendations.append(f"[{rec['category']}] {r}")
    
    return {
        "prediction": prediction_result,
        "probability": round(probability_diabetic, 4),
        "risk_score": risk_score,
        "confidence_level": confidence,
        "feature_contributions": contributions,
        "risk_factors": risk_factors if risk_factors else ["No significant risk factors detected"],
        "recommendations": flat_recommendations[:5],  # Top 5 recommendations
        "timestamp": datetime.now().isoformat()
    }

# --- 5. BATCH PREDICTION ENDPOINT ---
@app.post("/predict/batch", summary="Batch prediction for multiple patients")
def predict_batch(batch_input: BatchDiabetesInput):
    """Process multiple predictions in a single request."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    api_stats["predict_batch"] += 1
    
    results = []
    for patient_data in batch_input.patients:
        input_data = np.array([[patient_data.Pregnancies, patient_data.Glucose, 
                               patient_data.BloodPressure, patient_data.BMI, patient_data.Age]])
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        probability_diabetic = probabilities[1]
        
        results.append({
            "prediction": "Diabetic" if prediction == 1 else "Non-Diabetic",
            "probability": round(probability_diabetic, 4),
            "risk_score": calculate_risk_score(probability_diabetic),
            "input": patient_data.dict()
        })
    
    return {
        "total_patients": len(results),
        "predictions": results,
        "timestamp": datetime.now().isoformat()
    }

# --- 6. FEATURE IMPORTANCE ENDPOINT ---
@app.get("/feature_importance", response_model=List[FeatureImportanceOutput], summary="Get feature importance")
def get_feature_importance():
    """Returns feature importance scores."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    api_stats["feature_importance"] += 1
    
    importances = model.feature_importances_
    features = MODEL_METADATA["required_features"]
    
    importance_list = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(features, importances)
    ]
    
    importance_list.sort(key=lambda x: x["importance"], reverse=True)
    return importance_list

# --- 7. RISK ASSESSMENT ENDPOINT ---
@app.post("/risk_assessment", response_model=List[RiskAssessmentOutput], summary="Assess risk for each feature")
def assess_risk(data: DiabetesInput):
    """Analyze each feature and provide risk assessment."""
    api_stats["risk_assessment"] += 1
    
    assessments = []
    for feature in MODEL_METADATA["required_features"]:
        value = getattr(data, feature)
        risk_info = assess_feature_risk(feature, value)
        
        assessments.append({
            "feature": feature,
            "value": value,
            "status": risk_info["status"],
            "risk_level": risk_info["risk_level"],
            "normal_range": risk_info["normal_range"]
        })
    
    return assessments

# --- 8. WHAT-IF ANALYSIS ENDPOINT ---
@app.post("/what_if", response_model=WhatIfOutput, summary="What-if analysis")
def what_if_analysis(
    data: DiabetesInput,
    modified_feature: str,
    new_value: float
):
    """Analyze how changing a feature affects the prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    if modified_feature not in MODEL_METADATA["required_features"]:
        raise HTTPException(status_code=400, detail=f"Invalid feature. Must be one of: {MODEL_METADATA['required_features']}")
    
    api_stats["what_if"] += 1
    
    # Original prediction
    original_input = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.BMI, data.Age]])
    original_pred = model.predict(original_input)[0]
    original_prob = model.predict_proba(original_input)[0][1]
    
    # Modified prediction
    modified_data = data.dict()
    modified_data[modified_feature] = new_value
    modified_input = np.array([[modified_data["Pregnancies"], modified_data["Glucose"], 
                               modified_data["BloodPressure"], modified_data["BMI"], modified_data["Age"]]])
    modified_pred = model.predict(modified_input)[0]
    modified_prob = model.predict_proba(modified_input)[0][1]
    
    return {
        "original_prediction": "Diabetic" if original_pred == 1 else "Non-Diabetic",
        "original_probability": round(original_prob, 4),
        "modified_prediction": "Diabetic" if modified_pred == 1 else "Non-Diabetic",
        "modified_probability": round(modified_prob, 4),
        "probability_change": round(modified_prob - original_prob, 4)
    }

# --- 9. RECOMMENDATIONS ENDPOINT ---
@app.post("/recommendations", response_model=List[HealthRecommendation], summary="Get health recommendations")
def get_recommendations(data: DiabetesInput):
    """Get personalized health recommendations based on input data."""
    api_stats["recommendations"] += 1
    
    # Get prediction to determine risk level
    input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.BMI, data.Age]])
    prediction = model.predict(input_data)[0]
    prediction_result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    
    # Identify risk factors
    risk_factors = []
    if data.Glucose > 125:
        risk_factors.append("High glucose")
    if data.BMI > 30:
        risk_factors.append("Obesity")
    if data.BloodPressure > 90:
        risk_factors.append("High blood pressure")
    
    return generate_recommendations(data, prediction_result, risk_factors)

# --- 10. PREDICTION HISTORY ENDPOINT ---
@app.get("/history", summary="Get prediction history")
def get_prediction_history(limit: int = 10):
    """Retrieve recent prediction logs."""
    api_stats["history"] += 1
    return {
        "total_predictions": len(prediction_logs),
        "recent_predictions": prediction_logs[-limit:] if prediction_logs else [],
        "timestamp": datetime.now().isoformat()
    }

# --- 11. API STATISTICS ENDPOINT ---
@app.get("/stats", summary="Get API usage statistics")
def get_api_stats():
    """Returns API usage statistics."""
    total_calls = sum(api_stats.values())
    return {
        "total_api_calls": total_calls,
        "endpoint_usage": dict(api_stats),
        "total_predictions": len(prediction_logs),
        "timestamp": datetime.now().isoformat()
    }

# --- 12. FEATURE RANGES ENDPOINT ---
@app.get("/feature_ranges", summary="Get valid feature ranges")
def get_feature_ranges():
    """Returns valid ranges and normal values for all features."""
    api_stats["feature_ranges"] += 1
    return FEATURE_RANGES

# --- 13. MODEL PERFORMANCE ENDPOINT ---
@app.get("/model_performance", summary="Get model performance metrics")
def get_model_performance():
    """Returns comprehensive model performance metrics."""
    api_stats["model_performance"] += 1
    
    if not prediction_logs:
        return {
            "message": "No predictions yet to calculate performance",
            "metrics": MODEL_METADATA["metrics"]
        }
    
    # Calculate distribution of predictions
    diabetic_count = sum(1 for log in prediction_logs if log["prediction"] == "Diabetic")
    non_diabetic_count = len(prediction_logs) - diabetic_count
    
    # Calculate average probability
    avg_probability = sum(log["probability"] for log in prediction_logs) / len(prediction_logs)
    
    return {
        "training_metrics": MODEL_METADATA["metrics"],
        "prediction_distribution": {
            "diabetic": diabetic_count,
            "non_diabetic": non_diabetic_count,
            "total": len(prediction_logs)
        },
        "average_prediction_probability": round(avg_probability, 4),
        "timestamp": datetime.now().isoformat()
    }

# --- 14. CLEAR HISTORY ENDPOINT ---
@app.delete("/history", summary="Clear prediction history")
def clear_history():
    """Clear all prediction logs and statistics."""
    global prediction_logs, api_stats
    count = len(prediction_logs)
    prediction_logs = []
    api_stats = defaultdict(int)
    return {
        "message": f"Cleared {count} prediction logs and reset statistics",
        "timestamp": datetime.now().isoformat()
    }

# --- ROOT ENDPOINT ---
@app.get("/", summary="API information")
def root():
    """Returns API information and available endpoints."""
    return {
        "service": "Diabetes Prediction API - Enhanced",
        "version": MODEL_METADATA["version"],
        "status": "online",
        "endpoints": {
            "health": "/health",
            "model_info": "/model_info",
            "predict": "/predict",
            "predict_detailed": "/predict/detailed",
            "batch_predict": "/predict/batch",
            "feature_importance": "/feature_importance",
            "risk_assessment": "/risk_assessment",
            "what_if": "/what_if",
            "recommendations": "/recommendations",
            "history": "/history",
            "stats": "/stats",
            "feature_ranges": "/feature_ranges",
            "model_performance": "/model_performance"
        },
        "documentation": "/docs"
    }

# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import numpy as np
# from fastapi.middleware.cors import CORSMiddleware
# from typing import Dict, Any, List

# # --- MLOps METADATA ---
# MODEL_METADATA = {
#     "version": "1.0.0",
#     "algorithm": "Random Forest Classifier",
#     "training_date": "2024-03-15",
#     "metrics": {
#         "accuracy": 0.932,
#         "f1_score": 0.880,
#     },
#     "required_features": ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
# }
# # --- END MLOps METADATA ---

# app = FastAPI(
#     title="Diabetes Prediction API",
#     description="A FastAPI service for the MLOps Diabetes Prediction Model.",
#     version=MODEL_METADATA["version"]
# )

# # CORS configuration to allow your React front-end (or any client) to connect
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], 
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load the trained ML model
# try:
#     model = joblib.load("diabetes_model.pkl")
# except FileNotFoundError:
#     print("WARNING: 'diabetes_model.pkl' not found. API endpoints will fail.")
#     model = None # Set model to None if loading fails

# class DiabetesInput(BaseModel):
#     """Defines the input features for the prediction endpoint."""
#     Pregnancies: int
#     Glucose: float
#     BloodPressure: float
#     BMI: float
#     Age: int

# class PredictionOutput(BaseModel):
#     """Defines the structured output for the prediction endpoint."""
#     prediction: str
#     probability: float

# class FeatureImportanceOutput(BaseModel):
#     """Defines the structured output for the feature importance endpoint."""
#     feature: str
#     importance: float

# # --- 1. HEALTH CHECK ENDPOINT ---
# @app.get("/health", summary="Health check endpoint for Kubernetes probes")
# def health_check():
#     """Returns the status of the API and model loading."""
#     status = "OK" if model is not None else "ERROR: Model not loaded"
#     return {"status": status, "service": "diabetes-api"}

# # --- 2. MODEL INFO ENDPOINT (MLOPS FEATURE) ---
# @app.get("/model_info", response_model=Dict[str, Any], summary="Returns detailed metadata about the deployed model")
# def get_model_info():
#     """Returns the static metadata dictionary defined above."""
#     return MODEL_METADATA

# # --- 3. UPDATED PREDICT ENDPOINT (CONFIDENCE FEATURE) ---
# @app.post("/predict", response_model=PredictionOutput, summary="Predicts diabetes risk based on patient data")
# def predict(data: DiabetesInput):
#     if model is None:
#         raise HTTPException(status_code=503, detail="Model not loaded.")
        
#     # 1. Prepare input data for the model
#     input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.BMI, data.Age]])
    
#     # 2. Get the prediction result (0 or 1)
#     prediction = model.predict(input_data)[0]
    
#     # 3. Get the prediction probabilities for the positive class (Diabetic = index 1)
#     probabilities = model.predict_proba(input_data)[0]
#     probability_diabetic = probabilities[1] 

#     # 4. Map the integer prediction to a readable string
#     prediction_result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    
#     return {
#         "prediction": prediction_result,
#         "probability": probability_diabetic
#     }

# # --- 4. FEATURE IMPORTANCE ENDPOINT (EXPLAINABLE AI - XAI) ---
# @app.get("/feature_importance", response_model=List[FeatureImportanceOutput], summary="Returns the relative importance of each feature in the model")
# def get_feature_importance():
#     if model is None:
#         raise HTTPException(status_code=503, detail="Model not loaded.")
        
#     # Get importance scores from the RandomForest model
#     importances = model.feature_importances_
    
#     # Pair feature names with their importance scores
#     features = MODEL_METADATA["required_features"]
    
#     importance_list = [
#         {"feature": feature, "importance": float(importance)}
#         for feature, importance in zip(features, importances)
#     ]
    
#     # Sort the list by importance (highest first)
#     importance_list.sort(key=lambda x: x["importance"], reverse=True)
    
#     return importance_list

# # # # main.py
# # # from fastapi import FastAPI
# # # from pydantic import BaseModel
# # # import joblib
# # # import numpy as np
# # # from fastapi.middleware.cors import CORSMiddleware

# # # app = FastAPI()
# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins=["*"],  # You can later restrict this to ["http://localhost:5173"]
# # #     allow_credentials=True,
# # #     allow_methods=["*"],
# # #     allow_headers=["*"],
# # # )
# # # model = joblib.load("diabetes_model.pkl")

# # # class DiabetesInput(BaseModel):
# # #     Pregnancies: int
# # #     Glucose: float
# # #     BloodPressure: float
# # #     BMI: float
# # #     Age: int

# # # @app.get("/")
# # # def read_root():
# # #     return {"message": "Diabetes Prediction API is live"}

# # # @app.post("/predict")
# # # def predict(data: DiabetesInput):
# # #     input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.BMI, data.Age]])
# # #     prediction = model.predict(input_data)[0]
# # #     return {"diabetic": bool(prediction)}

# # from fastapi import FastAPI
# # from pydantic import BaseModel
# # import joblib
# # import numpy as np
# # from fastapi.middleware.cors import CORSMiddleware
# # from typing import Dict, Any

# # # --- MLOps METADATA ---
# # MODEL_METADATA = {
# #     "version": "1.0.0",
# #     "algorithm": "Random Forest Classifier",
# #     "training_date": "2024-03-15",
# #     "metrics": {
# #         "accuracy": 0.932,
# #         "f1_score": 0.880,
# #     },
# #     "required_features": ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
# # }
# # # --- END MLOps METADATA ---

# # app = FastAPI(
# #     title="Diabetes Prediction API",
# #     description="A FastAPI service for the MLOps Diabetes Prediction Model.",
# #     version=MODEL_METADATA["version"]
# # )

# # # CORS configuration to allow your React front-end (or any client) to connect
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"], 
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # Load the trained ML model
# # try:
# #     model = joblib.load("diabetes_model.pkl")
# # except FileNotFoundError:
# #     print("WARNING: 'diabetes_model.pkl' not found. API endpoints will fail.")
# #     model = None # Set model to None if loading fails

# # class DiabetesInput(BaseModel):
# #     """Defines the input features for the prediction endpoint."""
# #     Pregnancies: int
# #     Glucose: float
# #     BloodPressure: float
# #     BMI: float
# #     Age: int

# # class PredictionOutput(BaseModel):
# #     """Defines the structured output for the prediction endpoint."""
# #     prediction: str
# #     probability: float

# # # --- 1. HEALTH CHECK ENDPOINT ---
# # @app.get("/health", summary="Health check endpoint for Kubernetes probes")
# # def health_check():
# #     """Returns the status of the API and model loading."""
# #     status = "OK" if model is not None else "ERROR: Model not loaded"
# #     return {"status": status, "service": "diabetes-api"}

# # # --- 2. MODEL INFO ENDPOINT (MLOPS FEATURE) ---
# # @app.get("/model_info", response_model=Dict[str, Any], summary="Returns detailed metadata about the deployed model")
# # def get_model_info():
# #     """Returns the static metadata dictionary defined above."""
# #     return MODEL_METADATA

# # # --- 3. UPDATED PREDICT ENDPOINT (CONFIDENCE FEATURE) ---
# # @app.post("/predict", response_model=PredictionOutput, summary="Predicts diabetes risk based on patient data")
# # def predict(data: DiabetesInput):
# #     # 1. Prepare input data for the model
# #     input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.BMI, data.Age]])
    
# #     # 2. Get the prediction result (0 or 1)
# #     prediction = model.predict(input_data)[0]
    
# #     # 3. Get the prediction probabilities for the positive class (Diabetic = index 1)
# #     probabilities = model.predict_proba(input_data)[0]
# #     probability_diabetic = probabilities[1] 

# #     # 4. Map the integer prediction to a readable string
# #     prediction_result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    
# #     return {
# #         "prediction": prediction_result,
# #         "probability": probability_diabetic
# #     }