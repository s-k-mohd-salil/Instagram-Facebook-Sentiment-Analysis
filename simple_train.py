import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import re

# Simple preprocessing
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return text

# Load data
df = pd.read_csv('data/comments.csv')
df['processed'] = df['comment'].apply(preprocess)

# Vectorize
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['processed'])
y = df['sentiment']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train models
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)

nb = MultinomialNB()
nb.fit(X_train, y_train)

# Test
lr_pred = lr.predict(X_test)
nb_pred = nb.predict(X_test)

print(f"LR Accuracy: {accuracy_score(y_test, lr_pred):.3f}")
print(f"NB Accuracy: {accuracy_score(y_test, nb_pred):.3f}")

# Save
joblib.dump(lr, 'models/lr_model.pkl')
joblib.dump(nb, 'models/nb_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("Models saved!")