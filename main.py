from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Smart Customer Churn Predictor API")

pipeline = joblib.load("churn_pipeline.pkl")


class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, description="Months with the company")
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingServices: Literal["Single", "None", "Both"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., gt=0)


@app.get("/")
def health_check():
    return FileResponse('index.html')


@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        # Convert validated Pydantic input into a 1-row DataFrame
        input_df = pd.DataFrame([customer.model_dump()])

        # Run prediction (0 or 1) and churn probability (0.0 to 1.0)
        prediction = int(pipeline.predict(input_df)[0])
        probability = float(pipeline.predict_proba(input_df)[0][1])

        # Assign risk tier for frontend display
        if probability >= 0.70:
            risk_level = "High Risk"
        elif probability >= 0.40:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

        return {
            "churn_prediction": prediction,
            "churn_label": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability * 100, 2),
            "risk_level": risk_level,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))