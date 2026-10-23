from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from src.preprocessing import clean_text

# Initialize the FastAPI app
app = FastAPI(title="SMS Spam Classifier API", version="1.0.0")

# Load the trained model and vectorizer from the models/ folder
print("Loading model and vectorizer...")
model = joblib.load("models/spam_classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
print("Models loaded successfully!")

# Define the structure of the data we expect to receive
class SMSRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"message": "Congratulations! You have won a $1000 Walmart gift card. Click here to claim."}

@app.post("/predict")
def predict_spam(request: SMSRequest):
    # 1. Clean the input text using our professional preprocessing function
    cleaned_msg = clean_text(request.message)
    
    # 2. Vectorize the cleaned text (transform, do not fit!)
    vectorized_msg = vectorizer.transform([cleaned_msg])
    
    # 3. Make the prediction
    prediction = model.predict(vectorized_msg)[0]
    
    # 4. Get the confidence score (probability)
    probabilities = model.predict_proba(vectorized_msg)[0]
    classes = model.classes_
    confidence = float(probabilities[list(classes).index(prediction)])
    
    return {
        "input_message": request.message,
        "prediction": prediction,
        "confidence": round(confidence, 4)
    }