# Instagram & Facebook Sentiment Analysis

A complete Data Science mini project that analyzes social media comments from Instagram and Facebook, classifying them into Positive, Negative, or Neutral sentiments using Machine Learning.

## 📋 Project Overview

This project demonstrates a complete sentiment analysis pipeline including:
- Text preprocessing and cleaning
- Feature extraction using TF-IDF
- Machine learning model training (Logistic Regression & Naive Bayes)
- Model evaluation with accuracy scores and confusion matrices
- Interactive web application for real-time sentiment analysis
- Data visualizations including pie charts, bar graphs, and word clouds

## 🏗️ Project Structure

```
project/
│
├── data/
│   └── comments.csv              # Sample dataset with comments and sentiments
├── notebooks/
├── src/
│   ├── preprocessing.py          # Text cleaning and preprocessing functions
│   ├── train_model.py            # Model training and evaluation
│   └── predict.py                # Sentiment prediction functions
├── models/                       # Trained models (generated after training)
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- VS Code with Python extension installed

### Installation

1. **Clone or download the project** to your local machine

2. **Open the project in VS Code**

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (required for text processing):
   - Open VS Code terminal
   - Run Python and execute:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

### Running the Project

#### Step 1: Train the Models

First, train the machine learning models:

```bash
python src/train_model.py
```

This will:
- Load and preprocess the sample data
- Train Logistic Regression and Naive Bayes models
- Display accuracy scores and confusion matrices
- Save trained models to the `models/` directory

#### Step 2: Run the Web Application

Launch the Streamlit web app:

```bash
streamlit run app.py
```

The application will open in your default web browser with two main sections:

1. **Analyze Comment**: Enter any social media comment and get instant sentiment analysis
2. **Data Visualization**: View charts and graphs of the sentiment distribution

#### Step 3: Test the Prediction (Optional)

You can also test individual predictions from the command line:

```bash
python src/predict.py
```

## 📊 Features

### Machine Learning Models
- **Logistic Regression**: Effective for binary and multi-class classification
- **Naive Bayes**: Probabilistic classifier, works well with text data

### Text Processing
- Text cleaning (removing URLs, special characters)
- Stopword removal
- Tokenization and lemmatization
- TF-IDF vectorization

### Visualizations
- Pie chart showing sentiment distribution
- Bar graph for sentiment counts
- Word cloud of all comments
- Confusion matrices for model evaluation

### Web Application
- Clean, user-friendly interface
- Real-time sentiment analysis
- Confidence scores and probability distributions
- Model selection (Logistic Regression or Naive Bayes)

## 📈 Model Performance

The models are trained on a sample dataset and evaluated using:
- Accuracy score
- Precision, Recall, F1-score
- Confusion matrix visualization

## 🔧 Customization

### Adding More Data

To improve model performance, add more training data to `data/comments.csv`:
- Format: `comment,sentiment`
- Sentiments: `positive`, `negative`, `neutral`

### Modifying Preprocessing

Edit `src/preprocessing.py` to customize text cleaning and preprocessing steps.

### Training New Models

Modify `src/train_model.py` to experiment with different algorithms or hyperparameters.

## 📚 Libraries Used

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning algorithms
- **nltk**: Natural language processing
- **streamlit**: Web application framework
- **wordcloud**: Word cloud generation
- **joblib**: Model serialization

## 🤝 Contributing

Feel free to fork this project and add your own improvements!

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Analyzing! 📊✨**