# evaluations, confusion matrices

import io
import boto3
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from preprocessing import label_cols

s3 = boto3.client('s3')
BUCKET = 'secure-content-moderator-ranith-992382735117-ap-southeast-2-an'

def load_from_s3(key):
    """Load a joblib artifact from S3."""
    buffer = io.BytesIO()
    s3.download_fileobj(BUCKET, key, buffer)
    buffer.seek(0)
    return joblib.load(buffer)

if __name__ == '__main__':
    # Load test data and trained models from S3
    X_test_tfidf = load_from_s3('data/processed/X_test.joblib')
    y_test = load_from_s3('data/processed/y_test.joblib')

    models = {}
    for label in label_cols:
        models[label] = load_from_s3(f'models/{label}_model.joblib')

    # Generate confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, label in zip(axes.flat, label_cols):
        ConfusionMatrixDisplay.from_predictions(
            y_test[label],
            models[label].predict(X_test_tfidf),
            ax=ax
        )
        ax.set_title(label)

    plt.tight_layout()
    plt.savefig('docs/confusion_matrices.png')
    print('Saved confusion matrices to docs/confusion_matrices.png')