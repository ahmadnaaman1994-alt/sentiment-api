from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch





MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
APP_TITLE = "Sentiment Analysis API"
APP_DESCRIPTION = "API for multilingual sentiment analysis using HuggingFace model"
APP_VERSION = "1.0"

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

from pydantic import BaseModel, Field
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1)

def get_sentiment_label(predicted_class: int) -> str:
    if predicted_class <= 2:
        return "negative"
    elif predicted_class == 3:
        return "neutral"
    else:
        return "positive"

def analyze_sentiment(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item() + 1
    confidence = torch.softmax(logits, dim=1)[0][predicted_class - 1].item()

    return {
        "prediction": predicted_class,
        "label": get_sentiment_label(predicted_class),
        "confidence": confidence
           }

@app.post("/predict")
def predict_sentiment(req: SentimentRequest):
    result = analyze_sentiment(req.text)

    return {
        "text": req.text,
        **result
    }

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running!"}
