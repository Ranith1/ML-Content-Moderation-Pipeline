import json
import os
import re
import boto3
import joblib
import io
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')
BUCKET = os.environ['MODEL_BUCKET']
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

vectorizer = None
models = {}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_models():
    global vectorizer, models
    if vectorizer is None:
        buf = io.BytesIO()
        s3.download_fileobj(BUCKET, 'artifacts/vectorizer.joblib', buf)
        buf.seek(0)
        vectorizer = joblib.load(buf)
    for label in LABELS:
        if label not in models:
            buf = io.BytesIO()
            s3.download_fileobj(BUCKET, f'models/{label}_model.joblib', buf)
            buf.seek(0)
            models[label] = joblib.load(buf)

def handler(event, context):
    load_models()
    body = json.loads(event.get('body', '{}'))
    text = body.get('text', '')

    if not text or len(text) > 5000:
        return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid input'})}

    text_clean = clean_text(text)
    text_vec = vectorizer.transform([text_clean])

    predictions = {}
    for label in LABELS:
        prob = models[label].predict_proba(text_vec)[0][1]
        predictions[label] = {'score': round(float(prob), 4), 'flagged': bool(prob > 0.5)} 

    logger.info(json.dumps({'text_length': len(text), 'any_flagged': any(p['flagged'] for p in predictions.values())}))

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({'predictions': predictions})
    }