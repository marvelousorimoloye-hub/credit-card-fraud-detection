from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging

# ─── Configure logging ───
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── FastAPI app ───
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="XGBoost-based fraud prediction endpoint",
    version="1.0.0"
)

# ─── Rate limiting (5 requests per minute per IP) ───
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─── Load model & scaler ───
MODEL_PATH = "./models/xgboost_class_weights_best.joblib"
SCALER_PATH = "./models/scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model or scaler: {e}")
    raise RuntimeError("Model loading failed")

# ─── Input schema (exactly 30 features) ───
class Transaction(BaseModel):
    Time: float = Field(..., gt=0, description="Time in seconds")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount")

       
    class Config:
        json_schema_extra = {
            "example": {
                 "Time": 1.0,
                 "V1": -1.3598071336738,
                 "V2": -0.0727811733098497,
                 "V3": 2.53634673796914,
                 "V4": 1.37815522427443,
                 "V5": -0.338320769942518,
                 "V6": 0.462387777762292,
                 "V7": 0.239598554184543,
                 "V8": 0.0986979012610507,
                 "V9": 0.363786968606742,
                 "V10": 0.090794171978162,
                 "V11": -0.551599533260949,
                 "V12": -0.617800855762348,
                 "V13": -0.991389847235408,
                 "V14": -0.311169353699879,
                 "V15": 1.46817697209427,
                 "V16": -0.470400525259478,
                 "V17": 0.207971241929242,
                 "V18": 0.0257905801985591,
                 "V19": 0.403992960255733,
                 "V20": 0.251412098239705,
                 "V21": -0.0183067779441532,
                 "V22": 0.277837575558899,
                 "V23": -0.110473910188767,
                 "V24": 0.0669280749146731,
                 "V25": 0.128539358273735,
                 "V26": -0.189114843888824,
                 "V27": 0.133558376740563,
                 "V28": -0.0210530534538215,
                 "Amount": 149.62
            }
        } 
                
            
            

# ─── Prediction endpoint ───
@app.post("/predict", response_model=dict)
@limiter.limit("5/minute")
async def predict_fraud(transaction: Transaction, request: Request):
    try:
        # Convert input to numpy array (1, 30)
        features = np.array([[
            transaction.Time,
            transaction.V1, transaction.V2, transaction.V3, transaction.V4,
            transaction.V5, transaction.V6, transaction.V7, transaction.V8,
            transaction.V9, transaction.V10, transaction.V11, transaction.V12,
            transaction.V13, transaction.V14, transaction.V15, transaction.V16,
            transaction.V17, transaction.V18, transaction.V19, transaction.V20,
            transaction.V21, transaction.V22, transaction.V23, transaction.V24,
            transaction.V25, transaction.V26, transaction.V27, transaction.V28,
            transaction.Amount
        ]])

        # Scale using the same scaler from training
        features_scaled = scaler.transform(features)

        # Predict probability & class
        prob_fraud = model.predict_proba(features_scaled)[0, 1]
        is_fraud = model.predict(features_scaled)[0]

        return {
            "is_fraud": bool(is_fraud),
            "fraud_probability": round(float(prob_fraud), 6),
            "message": "Fraud detected" if is_fraud else "Transaction appears legitimate"
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ─── Health check ───

    @app.get("/health")
    def health_check():
     return {
        "status": "healthy",
        "model": "xgboost-tuned-class-weights",
        "version": "1.0",
        "uptime_seconds": int(time.time() - start_time)  # add global start_time = time.time() at top
    }
   