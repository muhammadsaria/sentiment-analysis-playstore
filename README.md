# 📱 Play Store Sentiment Analysis App



This project is a powerful Streamlit-based web app that analyzes user reviews from the Google Play Store using machine learning and natural language processing. It can predict review sentiment, generate visual insights, and even scrape and analyze live reviews from the Play Store.

## 🧠 Model Development

The core of this app is a powerful **LightGBM Classifier** trained to analyze review sentiment based on both text and additional numeric features.

### 🔬 Steps Involved:

1. **Exploratory Data Analysis (EDA):**
   - Reviewed distributions, sentiment imbalance, and feature correlations.
   - Cleaned noisy data and removed inconsistencies and nulls.

2. **Data Cleaning & Preprocessing:**
   - Applied Natural Language Processing (NLP) techniques:
     - Lowercasing, punctuation & URL removal
     - Stopword removal using `nltk`
     - Tokenization and whitespace normalization

3. **Feature Engineering:**
   - Extracted text features using **TF-IDF Vectorizer** (up to bi-grams)
   - Merged 37 numerical features from the dataset for improved performance

4. **Model Training:**
   - Used **LightGBMClassifier** with tuned hyperparameters
   - Handled class imbalance using stratified splits and cross-validation
   - Achieved:
     - 🎯 Accuracy: 96 %
     - 📉 ROC AUC Score: 98 %
    
5. **Model Export:**
   - Saved using `joblib` to load efficiently inside the Streamlit app

This hybrid approach combining **TF-IDF + numeric metadata + LightGBM** resulted in a high-performing and lightweight real-world deployable model.



## 📦 Features

✅ Sentiment Prediction (Positive / Negative)  
🌐 Auto-language detection and translation to English  
📁 Batch prediction via CSV upload  
📊 Confidence visual + Sentiment distribution charts  
☁️ Wordclouds for both positive and negative reviews  
📈 Trend analysis (if review dates available)  
🔍 Live Google Play App Review Scraper  
🧼 Text cleaning, stopword removal, TF-IDF vectorization  
📊 LightGBM model for classification  


## 🧪 Usage

### 🔍 Single Review
- Input a single Play Store review (in any language).
- App translates (if needed), cleans, predicts sentiment, and shows confidence.

### 📁 Batch Prediction
- Upload a CSV with a column named `Translated_Review`.
- (Optional: include `Review_Date` for trend visualization).
- Output: Cleaned text, sentiment predictions, download option.

### 🌐 Live App Scraper
- Enter a package name like `com.whatsapp`.
- App fetches live reviews, processes them, and visualizes results.


## 📚 Dependencies

Major Python packages used:

- `streamlit`
- `pandas`, `numpy`, `scipy`
- `nltk`
- `lightgbm`, `scikit-learn`, `joblib`
- `wordcloud`, `matplotlib`, `seaborn`
- `googletrans==4.0.0rc1`
- `google-play-scraper`

