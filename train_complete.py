#!/usr/bin/env python3
"""
Self-contained sentiment analysis model training script.
Run this script to train the models without dependencies.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
import os

def preprocess_text(text):
    """Simple text preprocessing"""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert to lowercase
    text = text.lower()
    return text

def main():
    print("🚀 Starting Sentiment Analysis Model Training")
    print("=" * 50)

    # Check if data file exists
    data_path = 'data/comments.csv'
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        return

    print("📂 Loading data...")
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded: {len(df)} rows")
        print(f"📊 Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    print("\n🔄 Preprocessing text...")
    df['processed_comment'] = df['comment'].apply(preprocess_text)
    print("✅ Text preprocessing completed")

    print("\n🔢 Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['processed_comment'])
    y = df['sentiment']
    print(f"✅ Vectorization completed: {X.shape[0]} samples, {X.shape[1]} features")

    print("\n✂️ Splitting data (70% train, 30% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"✅ Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"📊 Test set distribution: {y_test.value_counts().to_dict()}")

    # Train Logistic Regression
    print("\n🤖 Training Logistic Regression model...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    print("✅ Logistic Regression training completed")

    # Train Naive Bayes
    print("\n🤖 Training Naive Bayes model...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    print("✅ Naive Bayes training completed")

    # Evaluate models
    print("\n📊 Evaluating models...")

    lr_predictions = lr_model.predict(X_test)
    nb_predictions = nb_model.predict(X_test)

    lr_accuracy = accuracy_score(y_test, lr_predictions)
    nb_accuracy = accuracy_score(y_test, nb_predictions)

    print(f"📈 Logistic Regression Accuracy: {lr_accuracy:.3f}")
    print(f"📈 Naive Bayes Accuracy: {nb_accuracy:.3f}")
    print("\n📋 Classification Reports:")
    print("\nLogistic Regression:")
    print(classification_report(y_test, lr_predictions))

    print("\nNaive Bayes:")
    print(classification_report(y_test, nb_predictions))

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    print("\n💾 Saving models...")
    try:
        joblib.dump(lr_model, 'models/lr_model.pkl')
        joblib.dump(nb_model, 'models/nb_model.pkl')
        joblib.dump(vectorizer, 'models/vectorizer.pkl')
        print("✅ Models saved successfully to 'models/' directory")
    except Exception as e:
        print(f"❌ Error saving models: {e}")
        return

    print("\n🎉 Training completed successfully!")
    print("\n📝 Next steps:")
    print("1. Run the Streamlit app: streamlit run app.py")
    print("2. Test predictions: python src/predict.py")
    print("3. Explore the notebook: open notebooks/analysis.ipynb")

if __name__ == "__main__":
    main()