# model training

import boto3
import joblib
import io
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from preprocessing import label_cols


s3 = boto3.client('s3') 
BUCKET = 'secure-content-moderator-ranith-992382735117-ap-southeast-2-an'

# loading processed data from S3
def load_from_s3(key):
    """Load a joblib artifact from S3."""
    buffer = io.BytesIO() # file/buffer in RAM
    s3.download_fileobj(BUCKET, key, buffer) # download file from S3 into buffer 
    buffer.seek(0) # move pointer to start of buffer so that jobilb can read from beginning 
    return joblib.load(buffer) # deserialise object from buffer and return it   

if __name__ == '__main__':
    # Load train/test splits from S3
    # print('Loading data from S3...')
    X_train_tfidf = load_from_s3('data/processed/X_train.joblib')
    X_test_tfidf = load_from_s3('data/processed/X_test.joblib')
    y_train = load_from_s3('data/processed/y_train.joblib')
    y_test = load_from_s3('data/processed/y_test.joblib')

    models = {}
    results = {}

    # Train one binary classifier per label
    for label in label_cols:
        print(f'Training {label}...')
        
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',  # handles class imbalance
            solver='liblinear'
        )
        
        model.fit(X_train_tfidf, y_train[label])
        y_pred = model.predict(X_test_tfidf)
        
        models[label] = model
        results[label] = {
            'f1': f1_score(y_test[label], y_pred),
            'report': classification_report(y_test[label], y_pred)
        }
        
        print(f'{label} F1: {results[label]["f1"]:.3f}')
        print(results[label]['report'])

    # Save each trained model to S3
    for label, model in models.items():
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        s3.upload_fileobj(buffer, BUCKET, f'models/{label}_model.joblib')
        print(f'Saved {label} model to S3')
