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


class SentimentRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(req: SentimentRequest):
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item() + 1
    confidence = torch.softmax(logits, dim=1)[0][predicted_class - 1].item()

    return {
        "text": req.text,
        "prediction": predicted_class,
        "confidence": confidence
    }


@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running!"}
