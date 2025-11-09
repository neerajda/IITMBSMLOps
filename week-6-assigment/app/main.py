from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize the FastAPI app
app = FastAPI(
    title="Iris Flower Prediction API",
    description="Predict Iris species using a trained ML model",
    version="1.0.0"
)

# Define input schema
class InputData(BaseModel):
    features: list[float]

# Load your trained model (ensure model.pkl is in /app when container runs)
model = joblib.load("model.pkl")

@app.get("/")
def root():
    return {"message": "Welcome to the Iris Prediction API"}

@app.post("/predict")
def predict(data: InputData):
    """
    Accepts a list of 4 features and returns the predicted Iris class.
    Example:
    {
      "features": [5.1, 3.5, 1.4, 0.2]
    }
    """
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": str(prediction[0])}
