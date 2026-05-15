import os
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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

_loaded_models = None


def load_models():
    """
    Load the trained models and vectorizer.
    """
    global _loaded_models
    if _loaded_models is not None:
        return _loaded_models

    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for models in: {models_dir}")

    try:
        lr_model = joblib.load(os.path.join(models_dir, 'lr_model.pkl'))
        nb_model = joblib.load(os.path.join(models_dir, 'nb_model.pkl'))
        vectorizer = joblib.load(os.path.join(models_dir, 'vectorizer.pkl'))
        _loaded_models = (lr_model, nb_model, vectorizer)
        return _loaded_models
    except FileNotFoundError as e:
        print(f"Models not found: {e}")
        return None, None, None


def predict_sentiment(comment, model='lr'):
    """
    Predict sentiment for a given comment using the specified model.
    model: 'lr' for Logistic Regression, 'nb' for Naive Bayes
    """
    lr_model, nb_model, vectorizer = load_models()
    if lr_model is None or vectorizer is None:
        return None

    # Preprocess the comment
    processed_comment = preprocess_text(comment)

    # Vectorize
    comment_vector = vectorizer.transform([processed_comment])

    if model == 'lr':
        model_obj = lr_model
    elif model == 'nb':
        model_obj = nb_model
    else:
        print("Invalid model. Choose 'lr' or 'nb'.")
        return None

    prediction = model_obj.predict(comment_vector)[0]
    probabilities = model_obj.predict_proba(comment_vector)[0]
    classes = list(model_obj.classes_)
    prob_map = dict(zip(classes, probabilities))

    return {
        'sentiment': prediction,
        'confidence': float(max(probabilities)),
        'probabilities': {
            'negative': prob_map.get('negative', 0.0),
            'neutral': prob_map.get('neutral', 0.0),
            'positive': prob_map.get('positive', 0.0)
        }
    }

def main():
    # Example usage
    test_comments = [
        "I love this new feature!",
        "This is terrible, I hate it.",
        "It's okay, nothing special."
    ]

    for comment in test_comments:
        result = predict_sentiment(comment, model='lr')
        if result:
            print(f"Comment: {comment}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("Probabilities:", result['probabilities'])
            print("-" * 50)

if __name__ == "__main__":
    main()