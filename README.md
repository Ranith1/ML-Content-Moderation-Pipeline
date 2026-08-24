# ML-Content-Moderation-Pipeline

## Overview
Secure content moderation pipeline that classifies text across six toxicity categories. Trained on 160k labelled Wikipedia documents, deployed on AWS lambda behind API gateway, with security meatures against adversarial evasion, model tampering and API abuse.

The project prioritises production and security: least-privilege IAM, encryption at rest and in transit, SHA-256 model integrity verification, adversarial input sanitisation, PII scrubbing and custom Cloudwatch observability.

## Tech stack

**ML**: Python 3.13, scikit-learn, pandas, numpy, matplotlib
**AWS**: Lambda (arm64/Graviton), API Gateway, Cloudwatch, IAM
**Tooling**: boto3, joblib, pytest

Lambda runs on arm64 Graviton. Dependencies are packages as a Lambda Layer built for linus/arm64.

## Running Locally
Requires Python 3.13, an AWS account and a Kaggle account.

```bash
git clone https://github.com/Ranith1/ML-Content-Moderation-Pipeline.git
cd ML-Content-Moderation-Pipeline

python -m venv venv
source venv/bin/activate          
pip install -r requirements.txt
```

**Dataset** Download train.csv.zip from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
Place it in data/raw/.

**S3 bucket** Create a bucket in ap-southeast-2 with block-public access on, SSE-S3 encryption and versioning enabled. Update the BUCKET constant in src/preprocessing.py, src/train.py, src/evaluate.py and scripts/verify_integrity.py.

**Run the pipeline**

```bash
python src/preprocessing.py        # clean, split, vectorise, upload to S3
python src/train.py                # train 6 classifiers, upload to S3
python src/evaluate.py             # generate confusion matrices
python scripts/verify_integrity.py # generate and upload SHA-256 manifest
```

**Run the tests**

```bash
pytest tests/ -v
```

## Testing
Paste the below API call into the terminal:

```bash
curl -X POST https://t3acn677t1.execute-api.ap-southeast-2.amazonaws.com/prod/predict \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: thtHQlWY2CiEUXcP8WGV77SewpsdWvY7PEYJ2c1d' \
  -d '{"text": "you are an idiot"}'
```
A probability and flag for each of the six toxicity labels should be outputted.

**Endpoint** POST /predict
**Auth** x-api-key header required

Sample inputs:

"the weather is nice" - Clean text

"you are an idiot" - Multi-label output: toxic, obscene and insult

"y0u 4r3 4n 1d10t" - Leetspeak evasion is decoded before classification.

"email me at random@example.com idiot" - Email is redacted before the request touches CloudWatch

{"text": 123} - Input validaiton rejects non-string types with a 400.

