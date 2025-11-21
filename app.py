from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import sys
from typing import List, Optional
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(__file__))
from src.features import create_features

app = FastAPI(title="ReconGuard API", description="Anomaly Detection for Transactions")

# --- Model & Artifact Loading ---
# In a real app, these paths would be env vars or config
MODEL_PATH = 'models/autoencoder.pth'
SCALER_PATH = 'models/scaler.pkl'

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Global variables for artifacts
model = None
scaler = None
feature_cols = ['amount', 'hour', 'time_of_day', 'merchant_avg_amount', 'user_24h_count', 'user_24h_sum', 'user_24h_mean']

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    
    # Mocking artifacts if they don't exist for the sake of the demo/API running
    # In production, you would raise an error if they are missing.
    if os.path.exists(SCALER_PATH) and os.path.exists(MODEL_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            # Assuming input_dim is known or derived from scaler
            # For this demo, we'll just initialize a dummy model if loading fails or for structure
            # In real scenario: input_dim = scaler.n_features_in_
            input_dim = len(feature_cols) 
            model = Autoencoder(input_dim, 16)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            model.eval()
            print("Artifacts loaded successfully.")
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            # Fallback for demo purposes only
            create_dummy_artifacts()
    else:
        print("Artifacts not found. Creating dummy artifacts for demo.")
        create_dummy_artifacts()

def create_dummy_artifacts():
    global model, scaler
    from sklearn.preprocessing import StandardScaler
    
    # Create a dummy scaler
    scaler = StandardScaler()
    # Fit on some random data to initialize
    dummy_data = np.random.rand(10, len(feature_cols))
    scaler.fit(dummy_data)
    
    # Create a dummy model
    model = Autoencoder(len(feature_cols), 16)
    model.eval()

# --- Pydantic Models ---

class Transaction(BaseModel):
    user_id: str
    merchant_id: str
    amount: float
    timestamp: str # ISO format string

class PredictionResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    reasons: List[str]

# --- Endpoints ---

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    global model, scaler
    
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        data = {
            'user_id': [transaction.user_id],
            'merchant_id': [transaction.merchant_id],
            'amount': [transaction.amount],
            'timestamp': [transaction.timestamp]
        }
        df = pd.DataFrame(data)
        
        # Feature Engineering
        # Note: In a real real-time system, we would need to fetch historical state (user history) 
        # from a feature store (e.g., Redis). 
        # For this stateless demo, we will approximate or assume the input carries enough info 
        # or we calculate features based on just this row (which will be limited).
        # To make create_features work, we might need to mock history or accept it.
        # For simplicity here, we will run create_features but it will have 0 for rolling stats if history is missing.
        
        df_features = create_features(df)
        
        # Prepare input vector
        X = df_features[feature_cols].values
        
        # Scale
        X_scaled = scaler.transform(X)
        
        # Inference
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            reconstruction = model(X_tensor)
            mse = np.mean(np.power(X_scaled - reconstruction.numpy(), 2), axis=1)[0]
            
        # Logic for anomaly and reasons
        # Threshold is arbitrary for demo
        threshold = 0.1 
        is_anomaly = mse > threshold
        
        reasons = []
        if is_anomaly:
            reasons.append("High reconstruction error")
            # Simple heuristic for reasons: check which feature contributed most to error
            error_per_feature = np.power(X_scaled - reconstruction.numpy(), 2)[0]
            max_error_idx = np.argmax(error_per_feature)
            reasons.append(f"Unusual value for {feature_cols[max_error_idx]}")

        return PredictionResponse(
            anomaly_score=float(mse),
            is_anomaly=bool(is_anomaly),
            reasons=reasons
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "ReconGuard API is running"}
