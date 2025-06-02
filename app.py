import pickle
import re
import streamlit as st
from nltk.stem.porter import PorterStemmer
import pandas as pd
from datetime import datetime
import plotly.express as px

# Initialize stemmer and define stopwords
stemmer = PorterStemmer()
stopwords = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being',
    'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't",
    'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during',
    'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't",
    'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here',
    "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i',
    "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's",
    'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no',
    'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our',
    'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd",
    "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that',
    "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
    "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this',
    'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't",
    'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's",
    'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom',
    'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll",
    "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'
])

def preprocessing_text(text):
    """
    Preprocess text for sentiment analysis
    """
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove numbers and special chars
    words = text.split()
    words = [word for word in words if word not in stopwords and len(word) > 3]
    words = [stemmer.stem(word) for word in words]
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_models():
    """
    Load the trained models with caching for better performance
    """
    try:
        with open('model/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open('model/random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return tfidf_vectorizer, model
    except FileNotFoundError as e:
        st.error(f"Model files not found: {str(e)}")
        st.error("Please ensure 'tfidf_vectorizer.pkl' and 'random_forest_model.pkl' are in the 'model/' directory")
        return None, None

def get_sentiment_color(sentiment):
    """
    Return color based on sentiment
    """
    colors = {
        'Positive': '#28a745',
        'Neutral': '#ffc107', 
        'Negative': '#dc3545',
        'Irrelevant': '#6c757d'
    }
    return colors.get(sentiment, '#6c757d')

def get_confidence_interpretation(probabilities):
    """
    Interpret prediction confidence
    """
    max_prob = max(probabilities)
    if max_prob > 0.7:
        return "High confidence"
    elif max_prob > 0.5:
        return "Moderate confidence"
    else:
        return "Low confidence"

# Configuration
SENTIMENT_MAP = {
    0: 'Irrelevant',
    1: 'Negative', 
    2: 'Neutral',
    3: 'Positive'
}

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Tool", 
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load models
tfidf_vectorizer, model = load_models()

if tfidf_vectorizer is None or model is None:
    st.stop()

# Main interface
st.title("🔍 Real-Time Sentiment Analysis")
st.markdown("---")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This tool analyzes the sentiment of text using:
    - **TF-IDF Vectorization** for text representation
    - **Random Forest** classifier for sentiment prediction
    - **Advanced text preprocessing** including stemming and stopword removal
    """)
    
    st.header("📊 Sentiment Categories")
    for idx, sentiment in SENTIMENT_MAP.items():
        color = get_sentiment_color(sentiment)
        st.markdown(f"<span style='color: {color}'>● **{sentiment}**</span>", unsafe_allow_html=True)
    
    st.header("💡 Tips")
    st.write("""
    - Enter complete sentences for better accuracy
    - Works best with reviews, tweets, or feedback
    - Minimum 10-15 words recommended
    - Random Forest provides probability estimates
    """)

# Main input area
st.subheader("Enter Text for Analysis")

# Text input options
input_method = st.radio(
    "Choose input method:",
    ["Single text", "Multiple texts (batch)"],
    horizontal=True
)

if input_method == "Single text":
    user_input = st.text_area(
        "Text Input", 
        placeholder="Enter your text here (e.g., product review, tweet, feedback)...",
        height=120
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        predict_button = st.button("🎯 Analyze Sentiment", type="primary", use_container_width=True)
    
    if predict_button:
        if not user_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing sentiment..."):
                try:
                    # Preprocess input text
                    processed_text = preprocessing_text(user_input)
                    
                    # Check if preprocessing left any text
                    if not processed_text.strip():
                        st.warning("⚠️ Text doesn't contain meaningful words for analysis.")
                    else:
                        # Vectorize
                        text_vectorized = tfidf_vectorizer.transform([processed_text])
                        
                        # Predict
                        prediction = model.predict(text_vectorized)[0]
                        
                        # Get prediction probabilities (Random Forest supports this)
                        try:
                            probabilities = model.predict_proba(text_vectorized)[0]
                            confidence = get_confidence_interpretation(probabilities)
                        except:
                            probabilities = None
                            confidence = "N/A"
                        
                        sentiment_label = SENTIMENT_MAP.get(prediction, "Unknown")
                        color = get_sentiment_color(sentiment_label)
                        
                        # Display results
                        st.markdown("### 📈 Analysis Results")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(
                                f"<h3 style='color: {color}'>Sentiment: {sentiment_label}</h3>", 
                                unsafe_allow_html=True
                            )
                            st.write(f"**Confidence Level:** {confidence}")
                            
                            # Show top probability
                            if probabilities is not None:
                                max_prob = max(probabilities)
                                st.write(f"**Prediction Probability:** {max_prob:.2%}")
                        
                        with col2:
                            # Create probability visualization
                            if probabilities is not None:
                                prob_df = pd.DataFrame({
                                    'Sentiment': [SENTIMENT_MAP[i] for i in range(len(probabilities))],
                                    'Probability': probabilities
                                })
                                
                                fig = px.bar(
                                    prob_df, 
                                    x='Sentiment', 
                                    y='Probability',
                                    color='Sentiment',
                                    color_discrete_map={
                                        'Positive': '#28a745',
                                        'Neutral': '#ffc107',
                                        'Negative': '#dc3545', 
                                        'Irrelevant': '#6c757d'
                                    },
                                    title="Prediction Probabilities"
                                )
                                fig.update_layout(
                                    height=300,
                                    showlegend=False,
                                    yaxis_title="Probability",
                                    xaxis_title="Sentiment"
                                )
                                fig.update_yaxes(range=[0, 1])
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Show detailed probabilities
                        if probabilities is not None:
                            with st.expander("📊 Detailed Probabilities"):
                                prob_data = []
                                for i, prob in enumerate(probabilities):
                                    prob_data.append({
                                        'Sentiment': SENTIMENT_MAP[i],
                                        'Probability': f"{prob:.4f}",
                                        'Percentage': f"{prob*100:.2f}%"
                                    })
                                
                                prob_df_detailed = pd.DataFrame(prob_data)
                                prob_df_detailed = prob_df_detailed.sort_values('Probability', ascending=False)
                                st.dataframe(prob_df_detailed, use_container_width=True, hide_index=True)
                        
                        # Show processed text for transparency
                        with st.expander("🔍 View Processed Text"):
                            st.code(processed_text)
                            st.caption("This is how your text appears after preprocessing (lowercasing, removing stopwords, stemming)")
                        
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")

else:  # Batch processing
    st.subheader("📝 Batch Analysis")
    
    # File upload option
    uploaded_file = st.file_uploader(
        "Upload a CSV file with a 'text' column",
        type=['csv'],
        help="CSV should have a column named 'text' containing the texts to analyze"
    )
    
    # Or manual input
    batch_text = st.text_area(
        "Or enter multiple texts (one per line):",
        placeholder="Text 1\nText 2\nText 3...",
        height=200
    )
    
    if st.button("🎯 Analyze All", type="primary"):
        texts_to_analyze = []
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' in df.columns:
                    texts_to_analyze = df['text'].dropna().tolist()
                else:
                    st.error("CSV file must contain a 'text' column")
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
        
        elif batch_text.strip():
            texts_to_analyze = [text.strip() for text in batch_text.split('\n') if text.strip()]
        
        if texts_to_analyze:
            results = []
            progress_bar = st.progress(0)
            
            for i, text in enumerate(texts_to_analyze):
                try:
                    processed_text = preprocessing_text(text)
                    if processed_text.strip():
                        text_vectorized = tfidf_vectorizer.transform([processed_text])
                        prediction = model.predict(text_vectorized)[0]
                        sentiment = SENTIMENT_MAP.get(prediction, "Unknown")
                        
                        # Get confidence for batch processing too
                        try:
                            probabilities = model.predict_proba(text_vectorized)[0]
                            confidence_score = max(probabilities)
                        except:
                            confidence_score = None
                        
                        results.append({
                            'Original Text': text[:100] + "..." if len(text) > 100 else text,
                            'Sentiment': sentiment,
                            'Confidence': f"{confidence_score:.2%}" if confidence_score else "N/A",
                            'Prediction Code': prediction
                        })
                    else:
                        results.append({
                            'Original Text': text[:100] + "..." if len(text) > 100 else text,
                            'Sentiment': 'Unable to process',
                            'Confidence': "N/A",
                            'Prediction Code': -1
                        })
                    
                    progress_bar.progress((i + 1) / len(texts_to_analyze))
                    
                except Exception as e:
                    results.append({
                        'Original Text': text[:100] + "..." if len(text) > 100 else text,
                        'Sentiment': f'Error: {str(e)}',
                        'Confidence': "N/A",
                        'Prediction Code': -1
                    })
            
            progress_bar.empty()
            
            # Display results
            results_df = pd.DataFrame(results)
            st.subheader("📊 Batch Analysis Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Summary statistics
            valid_results = results_df[results_df['Prediction Code'] >= 0]
            if not valid_results.empty:
                sentiment_counts = valid_results['Sentiment'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Summary Statistics")
                    st.write(f"**Total Texts Analyzed:** {len(valid_results)}")
                    st.write("**Sentiment Distribution:**")
                    for sentiment, count in sentiment_counts.items():
                        percentage = (count / len(valid_results)) * 100
                        st.write(f"• **{sentiment}:** {count} ({percentage:.1f}%)")
                
                with col2:
                    # Pie chart
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution",
                        color=sentiment_counts.index,
                        color_discrete_map={
                            'Positive': '#28a745',
                            'Neutral': '#ffc107',
                            'Negative': '#dc3545',
                            'Irrelevant': '#6c757d'
                        }
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("⚠️ Please provide texts to analyze either via file upload or manual input.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with Streamlit • Powered by Random Forest & NLTK"
    "</div>", 
    unsafe_allow_html=True
)