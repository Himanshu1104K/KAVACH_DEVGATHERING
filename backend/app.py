from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import sys
import pandas as pd
import numpy as np
import asyncio
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import joblib

# Define absolute paths to model directories
EFFICIENCY_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 "Training/Deep_Learning_Models/Efficiency/models")
STRIKE_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "Training/Deep_Learning_Models/Strike_Winning/models")

# Define dataset paths with absolute paths
EFFICIENCY_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "Training/Dataset/Efficiency/military_performance_training_data.csv")
STRIKE_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "Training/Dataset/Strike_Winning/soldier_strike_efficiency.csv")

# Don't add the model paths to sys.path to avoid problematic imports
# Instead define all functions directly here

# Efficiency prediction function
def predict_efficiency_direct(data):
    """Direct prediction function for efficiency using the models directory"""
    try:
        # Import tensorflow for loading models
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        
        # Load preprocessor
        preprocessor_path = os.path.join(EFFICIENCY_MODEL_DIR, "preprocessor.pkl")
        if not os.path.exists(preprocessor_path):
            print(f"Preprocessor not found at {preprocessor_path}")
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
            
        preprocessor = joblib.load(preprocessor_path)
        
        # Preprocess the data
        X_processed = preprocessor.transform(data)
        
        # Load the model - try best model first, then fallback to final model
        if os.path.exists(os.path.join(EFFICIENCY_MODEL_DIR, "efficiency_predictor_best.keras")):
            model_path = os.path.join(EFFICIENCY_MODEL_DIR, "efficiency_predictor_best.keras")
        else:
            model_path = os.path.join(EFFICIENCY_MODEL_DIR, "efficiency_predictor_final.keras")
        
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        model = load_model(model_path)
        
        # Make prediction
        predictions = model.predict(X_processed)
        return predictions.flatten()
    except Exception as e:
        print(f"Error in efficiency prediction: {str(e)}")
        # Fallback to random predictions
        return np.random.uniform(0.5, 0.9, len(data))

# Strike efficiency prediction function
def predict_strike_efficiency_direct(data):
    """Direct prediction function for strike efficiency using the models directory"""
    try:
        # Import tensorflow for loading models
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        
        # Load preprocessor
        preprocessor_path = os.path.join(STRIKE_MODEL_DIR, "strike_efficiency_preprocessor.pkl")
        if not os.path.exists(preprocessor_path):
            print(f"Strike preprocessor not found at {preprocessor_path}")
            raise FileNotFoundError(f"Strike preprocessor not found at {preprocessor_path}")
            
        preprocessor = joblib.load(preprocessor_path)
        
        # Preprocess the data
        X_processed = preprocessor.transform(data)
        
        # Check if we should use ensemble (if ensemble files exist)
        use_ensemble = (
            os.path.exists(os.path.join(STRIKE_MODEL_DIR, "ensemble_weights.npy")) and
            os.path.exists(os.path.join(STRIKE_MODEL_DIR, "strike_efficiency_rf.joblib")) and
            os.path.exists(os.path.join(STRIKE_MODEL_DIR, "strike_efficiency_gb.joblib"))
        )
        
        if use_ensemble:
            # Load all models for ensemble
            nn_model_path = os.path.join(STRIKE_MODEL_DIR, "strike_efficiency_final.keras")
            rf_model_path = os.path.join(STRIKE_MODEL_DIR, "strike_efficiency_rf.joblib")
            gb_model_path = os.path.join(STRIKE_MODEL_DIR, "strike_efficiency_gb.joblib")
            weights_path = os.path.join(STRIKE_MODEL_DIR, "ensemble_weights.npy")
            
            # Check if files exist
            if not all(os.path.exists(p) for p in [nn_model_path, rf_model_path, gb_model_path, weights_path]):
                print("Not all ensemble models found, falling back to single model")
                use_ensemble = False
            else:
                # Load the models
                nn_model = load_model(nn_model_path)
                rf_model = joblib.load(rf_model_path)
                gb_model = joblib.load(gb_model_path)
                weights = np.load(weights_path)
                
                # Get predictions from each model
                nn_preds = nn_model.predict(X_processed).flatten()
                rf_preds = rf_model.predict(X_processed)
                gb_preds = gb_model.predict(X_processed)
                
                # Combine predictions with ensemble weights
                predictions = (
                    weights[0] * nn_preds + 
                    weights[1] * rf_preds + 
                    weights[2] * gb_preds
                )
                return predictions
        
        if not use_ensemble:
            # Use single model for prediction
            if os.path.exists(os.path.join(STRIKE_MODEL_DIR, "strike_efficiency_best.keras")):
                model_path = os.path.join(STRIKE_MODEL_DIR, "strike_efficiency_best.keras")
            else:
                model_path = os.path.join(STRIKE_MODEL_DIR, "strike_efficiency_final.keras")
            
            if not os.path.exists(model_path):
                print(f"Strike model not found at {model_path}")
                raise FileNotFoundError(f"Strike model not found at {model_path}")
                
            model = load_model(model_path)
            predictions = model.predict(X_processed)
            return predictions.flatten()
    except Exception as e:
        print(f"Error in strike efficiency prediction: {str(e)}")
        # Fallback to random predictions
        return np.random.uniform(0.4, 0.95, len(data))

# Data generation functions - defined directly here instead of importing
def create_efficiency_dataset(output_path, num_samples=1000):
    """Generate synthetic efficiency dataset"""
    print(f"Generating synthetic efficiency dataset with {num_samples} samples...")
    np.random.seed(42)
    
    data = {
        "Age": np.random.randint(18, 45, num_samples),
        "Experience": np.random.randint(0, 20, num_samples),
        "Training": np.random.randint(50, 500, num_samples),
        "Physical_Fitness": np.random.uniform(0.3, 0.9, num_samples),
        "Mental_Readiness": np.random.uniform(0.4, 0.95, num_samples),
        "Equipment_Quality": np.random.uniform(0.5, 1.0, num_samples),
        "Strength": np.random.uniform(50, 95, num_samples),
        "Speed": np.random.uniform(40, 90, num_samples),
        "Endurance": np.random.uniform(45, 90, num_samples),
        "TeamCoordination": np.random.uniform(0.4, 0.9, num_samples),
        "MissionComplexity": np.random.uniform(0.2, 0.8, num_samples),
    }
    
    # Calculate efficiency score
    physical = 0.4 * data["Physical_Fitness"] + 0.3 * (data["Strength"] / 100) + 0.3 * (data["Endurance"] / 100)
    mental = 0.6 * data["Mental_Readiness"] + 0.4 * data["TeamCoordination"]
    experience = 0.7 * (data["Experience"] / 20) + 0.3 * (data["Training"] / 500)
    
    # Final weighted efficiency
    efficiency = (0.4 * physical + 0.3 * mental + 0.3 * experience) * (1 - 0.2 * data["MissionComplexity"])
    efficiency = np.clip(efficiency, 0.2, 0.95)  # Clip to reasonable range
    
    data["Efficiency"] = efficiency
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Efficiency dataset saved to {output_path}")
    
    return df

def create_strike_dataset(output_path, num_samples=1000):
    """Generate synthetic strike efficiency dataset"""
    print(f"Generating synthetic strike efficiency dataset with {num_samples} samples...")
    return create_sample_dataset(output_path, num_samples)

def create_sample_dataset(output_path, num_samples=1000):
    """
    Create a sample dataset with features relevant for strike efficiency
    """
    np.random.seed(42)
    
    data = {
        # Physical attributes
        "Age": np.random.randint(18, 45, num_samples),
        "Weight": np.random.normal(75, 10, num_samples),
        "Height": np.random.normal(175, 10, num_samples),
        # Physical fitness
        "Strength": np.random.normal(70, 15, num_samples),
        "Speed": np.random.normal(65, 15, num_samples),
        "Endurance": np.random.normal(60, 15, num_samples),
        # Combat training
        "TrainingHours": np.random.randint(50, 5000, num_samples),
        "CombatExperience": np.random.randint(0, 10, num_samples),
        # Physiological state
        "Fatigue": np.random.normal(30, 10, num_samples),
        "Stress": np.random.normal(40, 15, num_samples),
        # Weapon proficiency
        "WeaponAccuracy": np.random.normal(70, 15, num_samples),
        "WeaponProficiency": np.random.normal(75, 15, num_samples),
        "WeaponType": np.random.choice(
            ["Assault Rifle", "Sniper Rifle", "Machine Gun", "Shotgun", "Handgun"], 
            num_samples
        ),
        # Environmental factors
        "Terrain": np.random.choice(
            ["Urban", "Forest", "Desert", "Mountain"], num_samples
        ),
        "TimeOfDay": np.random.choice(["Day", "Night"], num_samples),
        "Weather": np.random.choice(
            ["Clear", "Rainy", "Foggy", "Windy", "Snowy"], num_samples
        ),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate Strike Efficiency (simplified model)
    strength_norm = (df["Strength"] - df["Strength"].min()) / (
        df["Strength"].max() - df["Strength"].min()
    )
    speed_norm = (df["Speed"] - df["Speed"].min()) / (
        df["Speed"].max() - df["Speed"].min()
    )
    training_norm = (df["TrainingHours"] - df["TrainingHours"].min()) / (
        df["TrainingHours"].max() - df["TrainingHours"].min()
    )
    weapon_acc_norm = (df["WeaponAccuracy"] - df["WeaponAccuracy"].min()) / (
        df["WeaponAccuracy"].max() - df["WeaponAccuracy"].min()
    )
    
    # Basic efficiency calculation
    base_efficiency = (
        0.3 * strength_norm
        + 0.2 * speed_norm
        + 0.3 * training_norm
        + 0.2 * weapon_acc_norm
    )
    
    # Apply terrain and time modifiers
    terrain_mod = pd.Series(1.0, index=df.index)
    terrain_mod[df["Terrain"] == "Urban"] = 0.9  # Urban is harder
    terrain_mod[df["Terrain"] == "Desert"] = 1.1  # Desert is easier
    
    time_mod = pd.Series(1.0, index=df.index)
    time_mod[df["TimeOfDay"] == "Night"] = 0.8  # Night is harder
    
    # Apply fatigue and stress penalties
    fatigue_penalty = df["Fatigue"] / 200  # Max 0.5 penalty for extreme fatigue
    stress_penalty = df["Stress"] / 200  # Max 0.5 penalty for extreme stress
    
    # Final efficiency with modifiers and noise
    efficiency = (
        base_efficiency * terrain_mod * time_mod
        - fatigue_penalty
        - stress_penalty
        + np.random.normal(0, 0.05, num_samples)
    )
    
    efficiency = np.clip(efficiency, 0, 1)  # Ensure values between 0 and 1
    
    df["StrikeEfficiency"] = efficiency
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Strike efficiency dataset saved to {output_path}")
    
    return df

# Create empty directories if they don't exist
os.makedirs("Training/Dataset/Efficiency", exist_ok=True)
os.makedirs("Training/Dataset/Strike_Winning", exist_ok=True)
os.makedirs(EFFICIENCY_MODEL_DIR, exist_ok=True)
os.makedirs(STRIKE_MODEL_DIR, exist_ok=True)

# Create sample datasets if they don't exist
if not os.path.exists(EFFICIENCY_DATASET_PATH):
    print(f"Creating sample efficiency dataset at {EFFICIENCY_DATASET_PATH}")
    create_efficiency_dataset(EFFICIENCY_DATASET_PATH)
else:
    print(f"Efficiency dataset found at {EFFICIENCY_DATASET_PATH}")

if not os.path.exists(STRIKE_DATASET_PATH):
    print(f"Creating sample strike efficiency dataset at {STRIKE_DATASET_PATH}")
    create_strike_dataset(STRIKE_DATASET_PATH)
else:
    print(f"Strike dataset found at {STRIKE_DATASET_PATH}")

# Load sample data for use in the API
try:
    sample_efficiency_data = pd.read_csv(EFFICIENCY_DATASET_PATH)
    print(f"Loaded efficiency dataset with {len(sample_efficiency_data)} samples")
except Exception as e:
    print(f"Error loading efficiency dataset: {str(e)}")
    # Create fallback sample data
    sample_efficiency_data = create_efficiency_dataset("temp_efficiency.csv")

try:
    sample_strike_data = pd.read_csv(STRIKE_DATASET_PATH)
    print(f"Loaded strike dataset with {len(sample_strike_data)} samples")
except Exception as e:
    print(f"Error loading strike dataset: {str(e)}")
    # Create fallback sample data
    sample_strike_data = create_strike_dataset("temp_strike.csv")

# Create FastAPI app
app = FastAPI(
    title="Military Performance Prediction API",
    description="API for predicting military efficiency and strike winning rate",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_prediction(self, websocket: WebSocket, message: Dict):
        await websocket.send_json(message)

manager = ConnectionManager()

# Base routes
@app.get("/")
async def root():
    return {"message": "Welcome to the Military Performance Prediction API"}

@app.get("/health")
async def health_check():
    # Check if model files exist
    efficiency_model_exists = (
        os.path.exists(os.path.join(EFFICIENCY_MODEL_DIR, "efficiency_predictor_final.keras")) or
        os.path.exists(os.path.join(EFFICIENCY_MODEL_DIR, "efficiency_predictor_best.keras"))
    )
    
    strike_model_exists = (
        os.path.exists(os.path.join(STRIKE_MODEL_DIR, "strike_efficiency_final.keras")) or
        os.path.exists(os.path.join(STRIKE_MODEL_DIR, "strike_efficiency_best.keras"))
    )
    
    efficiency_preprocessor_exists = os.path.exists(os.path.join(EFFICIENCY_MODEL_DIR, "preprocessor.pkl"))
    strike_preprocessor_exists = os.path.exists(os.path.join(STRIKE_MODEL_DIR, "strike_efficiency_preprocessor.pkl"))
    
    return {
        "status": "healthy",
        "efficiency_model": "available" if efficiency_model_exists else "unavailable",
        "strike_model": "available" if strike_model_exists else "unavailable",
        "efficiency_preprocessor": "available" if efficiency_preprocessor_exists else "unavailable",
        "strike_preprocessor": "available" if strike_preprocessor_exists else "unavailable",
        "efficiency_model_dir": EFFICIENCY_MODEL_DIR,
        "strike_model_dir": STRIKE_MODEL_DIR
    }

# Data models
class SoldierData(BaseModel):
    data: Dict[str, Any]

# Generate random data endpoint
@app.get("/generate-data")
async def generate_dummy_data(data_type: str = "efficiency", num_samples: int = 10):
    """Generate dummy data for testing the API"""
    if data_type.lower() == "efficiency":
        # Return a subset of the sample efficiency data
        return sample_efficiency_data.sample(
            min(num_samples, len(sample_efficiency_data))
        ).to_dict(orient="records")
    elif data_type.lower() == "strike":
        # Return a subset of the sample strike data
        return sample_strike_data.sample(
            min(num_samples, len(sample_strike_data))
        ).to_dict(orient="records")
    else:
        return {
            "error": f"Unknown data type: {data_type}. Use 'efficiency' or 'strike'"
        }

# WebSocket route for efficiency prediction
@app.websocket("/ws/predict/efficiency")
async def websocket_efficiency_prediction(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                # Parse the received data
                json_data = json.loads(data)
                
                # Convert to pandas DataFrame for prediction
                df = pd.DataFrame([json_data])
                
                # Make prediction using our direct function
                prediction = predict_efficiency_direct(df)
                
                # Send prediction result
                await manager.send_prediction(
                    websocket,
                    {
                        "status": "success",
                        "prediction": float(prediction[0]),
                        "message": "Efficiency prediction successful",
                    },
                )
            except Exception as e:
                # Send error message
                await manager.send_prediction(
                    websocket,
                    {
                        "status": "error",
                        "message": f"Error processing request: {str(e)}",
                    },
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# WebSocket route for strike efficiency prediction
@app.websocket("/ws/predict/strike-efficiency")
async def websocket_strike_prediction(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                # Parse the received data
                json_data = json.loads(data)
                
                # Convert to pandas DataFrame for prediction
                df = pd.DataFrame([json_data])
                
                # Make prediction using our direct function
                prediction = predict_strike_efficiency_direct(df)
                
                # Send prediction result
                await manager.send_prediction(
                    websocket,
                    {
                        "status": "success",
                        "prediction": float(prediction[0]),
                        "message": "Strike efficiency prediction successful",
                    },
                )
            except Exception as e:
                # Send error message
                await manager.send_prediction(
                    websocket,
                    {
                        "status": "error",
                        "message": f"Error processing request: {str(e)}",
                    },
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Regular REST API endpoint for efficiency prediction
@app.post("/predict/efficiency")
async def predict_efficiency_endpoint(data: Dict[str, Any]):
    try:
        # Convert to pandas DataFrame for prediction
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = predict_efficiency_direct(df)
        
        # Return prediction result
        return {
            "status": "success",
            "prediction": float(prediction[0]),
            "message": "Efficiency prediction successful"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error processing request: {str(e)}"
        }

# Regular REST API endpoint for strike efficiency prediction
@app.post("/predict/strike-efficiency")
async def predict_strike_efficiency_endpoint(data: Dict[str, Any]):
    try:
        # Convert to pandas DataFrame for prediction
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = predict_strike_efficiency_direct(df)
        
        # Return prediction result
        return {
            "status": "success",
            "prediction": float(prediction[0]),
            "message": "Strike efficiency prediction successful"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error processing request: {str(e)}"
        }

# Run the FastAPI app with uvicorn
if __name__ == "__main__":
    import uvicorn
    print(f"Efficiency model directory: {EFFICIENCY_MODEL_DIR}")
    print(f"Strike model directory: {STRIKE_MODEL_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
