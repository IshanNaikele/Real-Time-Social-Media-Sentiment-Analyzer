# 🔍 Real-Time Sentiment Analysis Tool

A powerful web-based sentiment analysis application built with Streamlit that uses machine learning to classify text sentiment . The tool supports both single text analysis and batch processing with intuitive visualizations and detailed probability breakdowns.

## ✨ Features

- **Sentiment Analysis**: Instant classification of text into Positive, Negative, Neutral, or Irrelevant categories
- **Batch Processing**: Analyze multiple texts at once via CSV upload or manual input
- **Interactive Visualizations**: Probability distributions and sentiment breakdowns with Plotly charts
- **Advanced Text Preprocessing**: Includes stemming, stopword removal, and HTML tag cleaning
- **Confidence Scoring**: Get prediction confidence levels and detailed probability breakdowns
- **User-friendly Interface**: Clean, intuitive Streamlit interface with helpful tips and information

## 🎯 Sentiment Categories

- **Positive** 🟢: Favorable, happy, or supportive sentiment
- **Negative** 🔴: Unfavorable, sad, or critical sentiment  
- **Neutral** 🟡: Balanced or factual sentiment
- **Irrelevant** ⚪: Off-topic or unrelated content

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: Random Forest Classifier with TF-IDF Vectorization
- **Text Processing**: NLTK (Porter Stemmer, stopword removal)
- **Visualization**: Plotly Express
- **Data Handling**: Pandas
- **Model Persistence**: Pickle

## 📋 Prerequisites

- Python 3.7+
- Required Python packages (see requirements.txt)
- Pre-trained models (TF-IDF vectorizer and Random Forest model)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/IshanNaikele/sentiment-analysis-tool.git
   cd sentiment-analysis-tool
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the model files**
   - Create a `model/` directory in the project root
   - Place your trained models in the `model/` directory:
     - `tfidf_vectorizer.pkl`
     - `random_forest_model.pkl`

## 📦 Requirements

Create a `requirements.txt` file with the following dependencies:

```
streamlit>=1.28.0
pandas>=1.5.0
nltk>=3.8
plotly>=5.15.0
scikit-learn>=1.3.0
numpy>=1.24.0
```

## 🏃‍♂️ Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Start analyzing!** Enter text or upload CSV files for sentiment analysis

## 📁 Project Structure

```
sentiment-analysis-tool/
│
├── app.py                    # Main Streamlit application
├── model/                    # Directory for trained models
│   ├── tfidf_vectorizer.pkl  # TF-IDF vectorizer model
│   └── random_forest_model.pkl # Random Forest classifier
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
└── .gitignore               # Git ignore file
```

## 🎮 Usage

### Single Text Analysis
1. Select "Single text" input method
2. Enter your text in the text area
3. Click "🎯 Analyze Sentiment"
4. View results including sentiment classification, confidence level, and probability breakdown

### Batch Processing
1. Select "Multiple texts (batch)" input method
2. Either:
   - Upload a CSV file with a 'text' column, or
   - Enter multiple texts manually (one per line)
3. Click "🎯 Analyze All"
4. View comprehensive results and download as CSV

### CSV File Format
For batch processing via CSV upload, ensure your file has a column named 'text':
```csv
text
"This product is amazing!"
"I hate waiting in long lines."
"The weather is sunny today."
```

## 🔧 Model Training

The application expects pre-trained models in the `model/` directory. To train your own models:

1. **Prepare your dataset** with text and sentiment labels (0: Irrelevant, 1: Negative, 2: Neutral, 3: Positive)
2. **Preprocess the text** using the same preprocessing function
3. **Train TF-IDF vectorizer** and **Random Forest classifier**
4. **Save models** using pickle in the `model/` directory

Example training script structure:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Train your models here
# ...

# Save models
with open('model/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
    
with open('model/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
```

## 🎨 Customization

### Adding New Sentiment Categories
1. Update the `SENTIMENT_MAP` dictionary in `app.py`
2. Modify the `get_sentiment_color()` function for new colors
3. Retrain your model with the new categories

### Modifying Text Preprocessing
The `preprocessing_text()` function can be customized to:
- Add more aggressive cleaning
- Include different stopwords
- Use different stemming algorithms
- Add lemmatization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📊 Performance Tips

- For better accuracy, use complete sentences with at least 10-15 words
- The tool works best with reviews, social media posts, and feedback text
- Batch processing is optimized for large datasets
- Models are cached for improved performance

## 🐛 Troubleshooting

### Common Issues

1. **Model files not found**
   - Ensure `tfidf_vectorizer.pkl` and `random_forest_model.pkl` are in the `model/` directory
   - Check file permissions

2. **NLTK data missing**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

3. **Memory issues with large batches**
   - Process files in smaller chunks
   - Increase system memory allocation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

[Ishan Naikele](https://github.com/IshanNaikele)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Machine learning powered by [scikit-learn](https://scikit-learn.org/)
- Text processing with [NLTK](https://www.nltk.org/)
- Visualizations by [Plotly](https://plotly.com/)

## 📈 Future Enhancements

- [ ] Support for more languages
- [ ] Real-time model retraining
- [ ] Integration with social media APIs
- [ ] Advanced text preprocessing options
- [ ] Model performance metrics dashboard
- [ ] Export to different formats (JSON, Excel)

---

⭐ If you found this project helpful, please give it a star on GitHub!
