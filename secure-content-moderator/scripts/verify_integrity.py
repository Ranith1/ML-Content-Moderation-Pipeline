import hashlib
import boto3
import json
import io

s3 = boto3.client('s3')
BUCKET = 'secure-content-moderator-ranith-992382735117-ap-southeast-2-an'

ARTIFACTS = [
    'artifacts/vectorizer.joblib',
    'models/toxic_model.joblib',
    'models/severe_toxic_model.joblib',
    'models/obscene_model.joblib',
    'models/threat_model.joblib',
    'models/insult_model.joblib',
    'models/identity_hate_model.joblib',
]


# download file from S3 and compute SHA-256 hash.
def compute_sha256(bucket, key):
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    buf.seek(0)
    return hashlib.sha256(buf.read()).hexdigest()

if __name__ == '__main__':
    manifest = {}
    for artifact in ARTIFACTS:
        print(f'Hashing {artifact}...')
        manifest[artifact] = compute_sha256(BUCKET, artifact)
        print(f'  {manifest[artifact]}')
    
    # Save manifest locally
    with open('models/manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Upload manifest to S3
    s3.upload_file('models/manifest.json', BUCKET, 'models/manifest.json')
    print('Manifest saved to S3')