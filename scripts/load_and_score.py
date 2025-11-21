import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import sys

# Add src to path so we can import features
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features import create_features

# Define the Autoencoder class (must match the one used during training)
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid() # Assuming input data is normalized to [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_model(model_path, input_dim, encoding_dim=16):
    model = Autoencoder(input_dim, encoding_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def main():
    # Paths
    DATA_PATH = 'data/transactions.csv' # Placeholder path
    MODEL_PATH = 'models/autoencoder.pth' # Placeholder path
    SCALER_PATH = 'models/scaler.pkl' # Placeholder path
    OUTPUT_PATH = 'scored_transactions.csv'

    # Check if files exist
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return
    if not os.path.exists(SCALER_PATH):
        print(f"Error: {SCALER_PATH} not found.")
        return

    # Load data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Feature Engineering
    print("Generating features...")
    df_features = create_features(df)
    
    # Select features for model (exclude non-numeric or ID columns)
    # Assuming these are the columns used during training
    feature_cols = ['amount', 'hour', 'time_of_day', 'merchant_avg_amount', 'user_24h_count', 'user_24h_sum', 'user_24h_mean']
    X = df_features[feature_cols].values
    
    # Load Scaler
    print("Loading scaler...")
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)
    
    # Load Model
    print("Loading model...")
    input_dim = X_scaled.shape[1]
    model = load_model(MODEL_PATH, input_dim)
    
    # Inference
    print("Scoring transactions...")
    X_tensor = torch.FloatTensor(X_scaled)
    with torch.no_grad():
        reconstructions = model(X_tensor)
        mse = np.mean(np.power(X_scaled - reconstructions.numpy(), 2), axis=1)
        
    df_features['reconstruction_error'] = mse
    df_features['anomaly_score'] = mse # Simple alias
    
    # Save results
    print(f"Saving results to {OUTPUT_PATH}...")
    df_features.to_csv(OUTPUT_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
