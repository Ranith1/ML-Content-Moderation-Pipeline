# text cleaning, TF-IDF 

import re
import boto3
import joblib
import io
import pandas as pd 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading data
data = pd.read_csv('../data/raw/train.csv.zip')

 
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


# Train/Test Split - splitting data into training and testing sets 

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
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


