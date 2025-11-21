# ReconGuard

ReconGuard is an advanced anomaly detection system for financial transactions using Autoencoders. It detects anomalies by learning the normal patterns of user behavior and flagging transactions with high reconstruction errors.

## Features

- **Feature Engineering**: 
    - Rolling 24-hour user statistics (count, sum, mean).
    - Merchant average transaction amounts (expanding mean).
    - Time-of-day bucketing (Night, Morning, Afternoon, Evening).
- **Deep Learning Model**: Autoencoder-based anomaly detection.
- **Real-time Inference**: FastAPI backend for scoring transactions on the fly.
- **Explainability**: Provides reasons for anomalies (e.g., "High reconstruction error", "Unusual value for amount").
- **Frontend Demo**: React-based UI for testing the API.
- **Analysis**: Jupyter notebooks for EDA and model evaluation.

## Project Structure

```
ReconGuard/
├── src/
│   └── features.py          # Feature engineering logic
├── scripts/
│   └── load_and_score.py    # Batch inference script
├── notebooks/
│   └── demo_skeleton.ipynb  # EDA and analysis notebook
├── frontend/
│   └── src/
│       └── components/
│           └── ReconGuardDemo.jsx # React demo component
├── app.py                   # FastAPI backend
└── README.md                # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js (for frontend)

### Installation

1. Clone the repository.
2. Install Python dependencies:
   ```bash
   pip install pandas numpy torch scikit-learn fastapi uvicorn jupyterlab matplotlib seaborn
   ```

### Running the API

1. Start the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```
   The API will be available at `http://localhost:8000`.

### Running the Frontend Demo

(Assuming a React app structure is fully set up around the component)
1. Navigate to `frontend/`.
2. Install dependencies: `npm install`.
3. Start the development server: `npm start`.

### Running the Batch Inference Script

1. Ensure you have data in `data/transactions.csv` (or update the path in the script).
2. Run the script:
   ```bash
   python scripts/load_and_score.py
   ```

### Running Notebooks

1. Start Jupyter Lab:
   ```bash
   jupyter lab
   ```
2. Open `notebooks/demo_skeleton.ipynb`.
