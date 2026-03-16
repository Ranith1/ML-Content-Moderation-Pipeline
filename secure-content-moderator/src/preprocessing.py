# text cleaning, TF-IDF 

import re
import boto3
import joblib
import io
import pandas as pd 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


 
# remove the stopwords detector/option later just have it for now to experiment with later 
# (tf-idf handles stopwrods)
def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Clean the input text by converting to lowercase, removing HTML tags, URLs, special characters and whitespace.
    Works on a single string. 
    
    Args:
        text (str): The input text to clean.
        remove_stopwords (bool): Whether to remove English stop words.
        
    Returns:
        str: The cleaned text.
    """
    text = text.lower() # convert to lowercase
    text = re.sub(r'<[^>]+>', '', text) # HTML tags
    text = re.sub(r'http\S+|www\S+', '', text) # URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text) # everything except a-zA-z i.e special characters
    text = re.sub(r'\s+', ' ', text).strip() # Extra whitespace

    # splits text into words and removes stop words if specified, then joins back into a string
    if remove_stopwords:
        text = ' '.join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)
        
    return text

if __name__ == "__main__":
    # Train/Test Split - splitting data into training and testing sets 

    # Loading data
    data = pd.read_csv('../data/raw/train.csv.zip')

    # creating a new column of cleaned text - applying clean_text function to each column of 'comment_text'
    data['comment_text_clean'] = data['comment_text'].apply(clean_text)

    # train_test_split function from sklearn splits the data into training and testing sets with 80% of the data used for training and 20% for testing. The random_state parameter ensures reproducibility of the split.
    # 'X' is the input features, 'Y' is the labels 
    X_train, X_test, y_train, y_test = train_test_split(
        data['comment_text_clean'],
        data[label_cols],
        test_size = 0.2, # 20% of data for testing, 80% for training 
        random_state = 1 

    )

    # TF-IDF Vectorization 
    vectorizer = TfidfVectorizer(
        max_features=50000, 
        ngram_range=(1, 2), 
        min_df=3, 
        max_df=0.9, 
        sublinear_tf=True )

    # use fit_transform on training data only, fit calculates rarity values for training data to carry over to testing data
    X_train_tfidf = vectorizer.fit_transform(X_train) # Fit on training data - converted into numbers
    X_test_tfidf = vectorizer.transform(X_test) # Transform test data


    s3 = boto3.client('s3')
    BUCKET = 'secure-content-moderator-ranith-992382735117-ap-southeast-2-an'

    buffer = io.BytesIO()
    joblib.dump(vectorizer, buffer)
    buffer.seek(0)
    s3.upload_fileobj(buffer, BUCKET, 'artifacts/vectorizer.joblib')

    # Save processed data
    # (similarly for train/test splits)

    # Save X_train
    buffer = io.BytesIO()
    joblib.dump(X_train, buffer)
    buffer.seek(0)
    s3.upload_fileobj(buffer, BUCKET, 'data/processed/X_train.joblib')

    # Save X_test
    buffer = io.BytesIO()
    joblib.dump(X_test, buffer)
    buffer.seek(0)
    s3.upload_fileobj(buffer, BUCKET, 'data/processed/X_test.joblib')

    # Save y_train
    buffer = io.BytesIO()
    joblib.dump(y_train, buffer)
    buffer.seek(0)
    s3.upload_fileobj(buffer, BUCKET, 'data/processed/y_train.joblib')

    # Save y_test
    buffer = io.BytesIO()
    joblib.dump(y_test, buffer)
    buffer.seek(0)
    s3.upload_fileobj(buffer, BUCKET, 'data/processed/y_test.joblib')