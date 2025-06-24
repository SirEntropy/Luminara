from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import joblib
import os

# Import your model class (adjust import as needed)
from src.models.crf import InvestmentCRF
from src.models.cliques import load_default_cliques
import yaml

app = FastAPI()

class InvestmentRequest(BaseModel):
    evidence: Dict[str, int]

# Load model config and instantiate model at startup (best practice)
MODEL_PATH = os.environ.get("MODEL_PATH", "results/models/synthetic/investment_crf.pkl")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config/model_config.yaml")

@app.on_event("startup")
def load_model():
    global crf_model
    if os.path.exists(MODEL_PATH):
        crf_model = joblib.load(MODEL_PATH)
    else:
        # Fallback: build a new model from config if not found
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        cliques, cardinalities = load_default_cliques(config)
        crf_model = InvestmentCRF(cliques, cardinalities)

@app.post("/predict")
def predict(request: InvestmentRequest):
    evidence = request.evidence
    try:
        proba = crf_model.predict(evidence)
        return {"probability_worthy": proba}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
