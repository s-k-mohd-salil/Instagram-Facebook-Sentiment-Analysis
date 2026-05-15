import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'comments.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def clean_text(text):
    """
    Clean the input text by removing special characters, URLs, and extra spaces.
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert to lowercase
    text = text.lower()
    return text

def remove_stopwords(text):
    """
    Remove stopwords from the text.
    """
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def tokenize_and_lemmatize(text):
    """
    Tokenize the text and apply lemmatization.
    """
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def preprocess_text(text):
    """
    Complete preprocessing pipeline: clean, remove stopwords, tokenize and lemmatize.
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    text = tokenize_and_lemmatize(text)
    return text

def load_and_preprocess_data(file_path):
    """
    Load data from CSV and preprocess the comments.
    """
    df = pd.read_csv(file_path)
    df['processed_comment'] = df['comment'].apply(preprocess_text)
    return df

def vectorize_text(texts):
    """
    Convert text data to TF-IDF vectors.
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def train_models(X_train, y_train):
    """
    Train Logistic Regression and Naive Bayes models.
    """
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)

    # Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    return lr_model, nb_model

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate the model and print accuracy and confusion matrix.
    """
    y_pred = model.predict(X_test)

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    #             xticklabels=['negative', 'neutral', 'positive'],
    #             yticklabels=['negative', 'neutral', 'positive'])
    # plt.title(f'Confusion Matrix - {model_name}')
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')
    # plt.show()

    return accuracy_score(y_test, y_pred)

def main():
    print("Starting training...")
    # Load and preprocess data
    df = load_and_preprocess_data(DATA_PATH)
    print(f"Data loaded: {len(df)} rows")
    print("Sentiment distribution:", df['sentiment'].value_counts().to_dict())

    # Prepare features and labels
    X, vectorizer = vectorize_text(df['processed_comment'])
    y = df['sentiment']
    print(f"Feature matrix shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    print("Test labels:", y_test.value_counts().to_dict())

    # Train models
    lr_model, nb_model = train_models(X_train, y_train)

    # Evaluate models
    lr_accuracy = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    nb_accuracy = evaluate_model(nb_model, X_test, y_test, "Naive Bayes")

    # Save models and vectorizer
    os.makedirs(MODELS_DIR, exist_ok=True)
    import joblib
    joblib.dump(lr_model, os.path.join(MODELS_DIR, 'lr_model.pkl'))
    joblib.dump(nb_model, os.path.join(MODELS_DIR, 'nb_model.pkl'))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'vectorizer.pkl'))

    print(f"\nModels and vectorizer saved to '{MODELS_DIR}' directory.")

    return lr_model, nb_model, vectorizer

if __name__ == "__main__":
    main()