from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI(title="Sentiment Analysis API",
              description="API for multilingual sentiment analysis using HuggingFace model",
              version="1.0")


model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


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
