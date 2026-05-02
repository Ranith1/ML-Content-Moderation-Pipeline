import json
import os
import re
import boto3
import joblib
import io
import logging
import hashlib
import time
cloudwatch = boto3.client('cloudwatch')

from input_sanitizer import sanitize_input
from pii_detector import detect_pii, scrub_pii

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
    manifest = load_manifest()
    
    if vectorizer is None:
        buf = io.BytesIO()
        s3.download_fileobj(BUCKET, 'artifacts/vectorizer.joblib', buf)
        if not verify_model_integrity(buf, manifest['artifacts/vectorizer.joblib']):
            logger.error('SECURITY: Vectorizer integrity check failed')
            raise ValueError('Model integrity check failed')
        vectorizer = joblib.load(buf)

    for label in LABELS:
        if label not in models:
            key = f'models/{label}_model.joblib'
            buf = io.BytesIO()
            s3.download_fileobj(BUCKET, key, buf)
            if not verify_model_integrity(buf, manifest[key]):
                logger.error(f'SECURITY: {label} model integrity check failed')
                raise ValueError(f'Model integrity check failed for {label}')
            models[label] = joblib.load(buf)

def handler(event, context):
    start_time = time.time()
    
    load_models()
    
    # validate request body exists
    raw_body = event.get('body', '')
    if not raw_body:
        logger.warning('Rejected request: empty body')
        return {'statusCode': 400, 'body': json.dumps({'error': 'Request body required'})}
    
    # validate body size (max 10KB)
    if len(raw_body) > 10000:
        logger.warning(f'Rejected request: body too large ({len(raw_body)} bytes)')
        return {'statusCode': 400, 'body': json.dumps({'error': 'Request body too large'})}
    
    # parse JSON
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        logger.warning('Rejected request: invalid JSON')
        return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid JSON'})}
    
    text = body.get('text', '')
    
    # validate text is a string
    if not isinstance(text, str):
        logger.warning(f'Rejected request: text is not a string (type: {type(text).__name__})')
        return {'statusCode': 400, 'body': json.dumps({'error': 'text must be a string'})}
    
    # validate text length
    if not text or len(text) > 5000:
        logger.warning(f'Rejected request: invalid text length ({len(text)})')
        return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid input'})}

    # sanitize for adversarial inputs (7A)
    sanitized = sanitize_input(text)
    if sanitized['evasion_detected']:
        logger.warning(f'Evasion attempt detected: unicode_normalized={sanitized["unicode_normalized"]}')
        emit_metric('EvasionAttempts', 1)

    # check and scrub PII
    pii_found = detect_pii(text)
    if pii_found:
        logger.warning(f'PII detected in request: {list(pii_found.keys())}')
        emit_metric('PIIDetected', 1)
        text = scrub_pii(text)

    # use sanitized text for classification
    text_clean = clean_text(sanitized['cleaned_text'])
    if not text_clean.strip():
        logger.warning('Rejected request: empty text after cleaning')
        return {'statusCode': 400, 'body': json.dumps({'error': 'Text is empty after cleaning'})}
    
    text_vec = vectorizer.transform([text_clean])

    predictions = {}
    for label in LABELS:
        prob = models[label].predict_proba(text_vec)[0][1]
        predictions[label] = {'score': round(float(prob), 4), 'flagged': bool(prob > 0.5)}

    # emit metrics
    elapsed_ms = (time.time() - start_time) * 1000
    emit_metric('PredictionLatency', elapsed_ms, 'Milliseconds')

    any_flagged = any(p['flagged'] for p in predictions.values())
    if any_flagged:
        emit_metric('FlaggedContent', 1)

    logger.info(json.dumps({
        'text_length': len(text),
        'any_flagged': any_flagged,
        'evasion_detected': sanitized['evasion_detected'],
        'pii_detected': bool(pii_found),
        'latency_ms': round(elapsed_ms, 2)
    }))

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({'predictions': predictions})
    }

# verify SHA-256 hash of model artifact
def verify_model_integrity(buf, expected_hash):
    buf.seek(0)
    actual_hash = hashlib.sha256(buf.read()).hexdigest()
    buf.seek(0)
    return actual_hash == expected_hash

# load model integrity manifest from S3.
def load_manifest():
    buf = io.BytesIO()
    s3.download_fileobj(BUCKET, 'models/manifest.json', buf)
    buf.seek(0)
    return json.loads(buf.read().decode('utf-8'))

# emit a custom metric to cloudwatch for monitoring purposes 
def emit_metric(name, value, unit='Count'):
    try:
        cloudwatch.put_metric_data(
            Namespace='ContentModerator',
            MetricData=[{
                'MetricName': name,
                'Value': value,
                'Unit': unit
            }]
        )
    except Exception as e:
        logger.error(f'Failed to emit metric {name}: {e}')