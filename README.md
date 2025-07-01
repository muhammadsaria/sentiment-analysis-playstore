# 📱 Play Store Sentiment Analysis App



This project is a powerful Streamlit-based web app that analyzes user reviews from the Google Play Store using machine learning and natural language processing. It can predict review sentiment, generate visual insights, and even scrape and analyze live reviews from the Play Store.

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

