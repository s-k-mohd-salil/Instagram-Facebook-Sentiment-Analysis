import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from src.predict import predict_sentiment
from src.preprocessing import preprocess_text

# Set page configuration
st.set_page_config(page_title="Sentiment Analysis", page_icon="📊", layout="wide")

# Title
st.title("📱 Instagram & Facebook Sentiment Analysis")
st.markdown("Analyze the sentiment of social media comments using Machine Learning!")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Analyze Comment", "Data Visualization"])

if page == "Analyze Comment":
    st.header("🔍 Analyze a Comment")

    # Input text area
    user_comment = st.text_area("Enter your comment:", height=100,
                               placeholder="Type a social media comment here...")

    # Model selection
    model_choice = st.selectbox("Choose Model:",
                               ["Logistic Regression", "Naive Bayes"],
                               index=0)

    model_map = {"Logistic Regression": "lr", "Naive Bayes": "nb"}

    # Analyze button
    if st.button("Analyze Sentiment", type="primary"):
        if user_comment.strip():
            with st.spinner("Analyzing..."):
                result = predict_sentiment(user_comment, model=model_map[model_choice])

            if result:
                # Display result
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Sentiment", result['sentiment'].capitalize())

                with col2:
                    st.metric("Confidence", f"{result['confidence']:.2%}")

                with col3:
                    # Sentiment emoji
                    if result['sentiment'] == 'positive':
                        st.write("😊")
                    elif result['sentiment'] == 'negative':
                        st.write("😞")
                    else:
                        st.write("😐")

                # Probabilities bar chart
                st.subheader("Probability Distribution")
                probs = result['probabilities']
                prob_df = pd.DataFrame({
                    'Sentiment': list(probs.keys()),
                    'Probability': list(probs.values())
                })

                st.bar_chart(prob_df.set_index('Sentiment'))

                # Processed text
                st.subheader("Processed Text")
                processed = preprocess_text(user_comment)
                st.write(f"**Original:** {user_comment}")
                st.write(f"**Processed:** {processed}")
            else:
                st.error("Models not found. Please train the models first by running train_model.py")
        else:
            st.warning("Please enter a comment to analyze.")

elif page == "Data Visualization":
    st.header("📊 Data Visualizations")

    # Load data
    try:
        df = pd.read_csv('data/comments.csv')

        # Sentiment distribution
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sentiment Distribution - Pie Chart")
            sentiment_counts = df['sentiment'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                   colors=['#ff9999','#66b3ff','#99ff99'])
            ax.axis('equal')
            st.pyplot(fig)

        with col2:
            st.subheader("Sentiment Distribution - Bar Chart")
            fig, ax = plt.subplots()
            sentiment_counts.plot(kind='bar', ax=ax, color=['#ff9999','#66b3ff','#99ff99'])
            ax.set_ylabel('Count')
            ax.set_xlabel('Sentiment')
            plt.xticks(rotation=0)
            st.pyplot(fig)

        # Word Cloud
        st.subheader("Word Cloud of Comments")
        all_text = ' '.join(df['comment'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        # Data table
        st.subheader("Sample Data")
        st.dataframe(df.head(10))

    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'data/comments.csv' exists.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Python, Streamlit, and Scikit-learn")