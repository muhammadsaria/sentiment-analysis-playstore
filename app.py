import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse
from wordcloud import WordCloud
from googletrans import Translator
from nltk.corpus import stopwords
from lightgbm import LGBMClassifier

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
translator = Translator()

# Load model and vectorizer
model = joblib.load("lightgbm_model.joblib")
tfidf = joblib.load("tfidf_vectorizer.joblib")

# -----------------------------
# Preprocessing
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])

def preprocess(text):
    return remove_stopwords(clean_text(text))

def auto_translate(text):
    detected = translator.detect(text).lang
    if detected != 'en':
        translated = translator.translate(text, dest='en').text
        return translated, detected
    return text, 'en'

# -----------------------------
# Visuals
# -----------------------------
def plot_probability(prob):
    st.subheader("\U0001F4CA Prediction Confidence")
    labels = ['Negative', 'Positive']
    fig, ax = plt.subplots()
    sns.barplot(x=labels, y=prob, palette="viridis", ax=ax)
    ax.set_ylim(0, 1)
    st.pyplot(fig)

def show_wordcloud(reviews, title):
    wc = WordCloud(width=600, height=400, background_color="white").generate(" ".join(reviews))
    st.subheader(title)
    st.image(wc.to_array())

def sentiment_trend_chart(df):
    df['Review_Date'] = pd.to_datetime(df['Review_Date'], errors='coerce')
    trend = df.groupby(['Review_Date', 'Sentiment_Label']).size().unstack(fill_value=0)
    st.subheader("\U0001F4C5 Sentiment Trend Over Time")
    st.line_chart(trend)

# -----------------------------
# Predict
# -----------------------------
def predict_sentiment(text):
    translated_text, lang = auto_translate(text)
    cleaned = preprocess(translated_text)
    X_text = tfidf.transform([cleaned])
    X_num = np.zeros((1, 37), dtype='float32')
    X_combined = scipy.sparse.hstack([X_text, X_num])
    proba = model.predict_proba(X_combined)[0]
    pred = model.predict(X_combined)[0]
    return pred, proba, cleaned, lang, X_text

# -----------------------------
# Layout
# -----------------------------
st.set_page_config(
    page_title="Sentiment Analysis - Play Store",
    layout="wide",
    page_icon="📱"
)
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
    font-family: 'Segoe UI', sans-serif;
}
.sidebar .sidebar-content {
    background-color: #e8eaf6;
}
</style>
<h1 style='text-align: center; color: #4CAF50;'>📱 ReviewPulse: Google Play Sentiment Explorer

</h1>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<h2 style='color:#6A5ACD;'>💡 Navigation</h2>
""", unsafe_allow_html=True)
option = st.sidebar.radio("Choose an option", [
    "🔍 Single Review",
    "📁 Batch Prediction (CSV)",
    "🌐 Live App Review Scraper"
])

# -----------------------------
# Single Review
# -----------------------------
if option == "\U0001F50D Single Review":
    review = st.text_area("✍️ Enter a Play Store Review:", "", height=150)
    import time
if st.button("Predict Sentiment"):
    with st.spinner("Analyzing sentiment..."):
        time.sleep(1)
        if not review.strip():
            st.warning("Please enter a review.")
        else:
            pred, proba, cleaned, lang, X_text = predict_sentiment(review)
            label = "Positive ✅" if pred == 1 else "Negative ❌"
            st.success(f"**Predicted Sentiment:** {label}")
            st.info(f"🌐 Language Detected: `{lang}`")
            st.info(f"🧼 Cleaned Text: `{cleaned}`")
            plot_probability(proba)



# -----------------------------
# Batch CSV Upload
# -----------------------------
elif option == "📁 Batch Prediction (CSV)":
    uploaded_file = st.file_uploader("Upload CSV with `Translated_Review` and optional `Review_Date`", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "Translated_Review" not in df.columns:
            st.error("CSV must contain a column named `Translated_Review`.")
        else:
            df["Translated"] = df["Translated_Review"].astype(str).apply(lambda x: auto_translate(x)[0])
            df["Clean_Review"] = df["Translated"].apply(preprocess)
            X_text = tfidf.transform(df["Clean_Review"])
            X_num = np.zeros((len(df), 37), dtype='float32')
            X_combined = scipy.sparse.hstack([X_text, X_num])
            df["Predicted"] = model.predict(X_combined)
            df["Sentiment_Label"] = df["Predicted"].apply(lambda x: "Positive" if x == 1 else "Negative")

            st.dataframe(df[["Translated_Review", "Sentiment_Label"]].head(20), use_container_width=True)
            st.subheader("📊 Sentiment Distribution")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x="Sentiment_Label", palette="Set2", ax=ax)
            st.pyplot(fig)

            if "Review_Date" in df.columns:
                sentiment_trend_chart(df)

            show_wordcloud(df[df["Predicted"] == 1]["Clean_Review"], "☁️ Positive Review WordCloud")
            show_wordcloud(df[df["Predicted"] == 0]["Clean_Review"], "☁️ Negative Review WordCloud")

            st.download_button("📥 Download Results", data=df.to_csv(index=False), file_name="predicted_sentiment.csv", mime="text/csv")

# -----------------------------
# Live App Review Scraper
# -----------------------------
elif option == "🌐 Live App Review Scraper":
    app_id = st.text_input("Enter App Package ID (e.g. com.whatsapp):")
    import time
if st.button("🔍 Fetch and Analyze Reviews"):
    with st.spinner("Fetching and analyzing reviews..."):
        time.sleep(1)
        if not app_id.strip():
            st.warning("Please enter a valid app package name.")
        else:
            from google_play_scraper import reviews
            try:
                with st.spinner("Fetching reviews..."):
                    result, _ = reviews(app_id, lang='en', country='us', count=100, filter_score_with=None)
                review_texts = [r['content'] for r in result]
                df = pd.DataFrame(review_texts, columns=['Translated_Review'])
                df["Translated"] = df["Translated_Review"].astype(str).apply(lambda x: auto_translate(x)[0])
                df["Clean_Review"] = df["Translated"].apply(preprocess)
                X_text = tfidf.transform(df["Clean_Review"])
                X_num = np.zeros((len(df), 37), dtype='float32')
                X_combined = scipy.sparse.hstack([X_text, X_num])
                df["Predicted"] = model.predict(X_combined)
                df["Sentiment_Label"] = df["Predicted"].apply(lambda x: "Positive" if x == 1 else "Negative")

                st.subheader("📊 Sentiment Distribution")
                fig, ax = plt.subplots()
                sns.countplot(data=df, x="Sentiment_Label", palette="Set2", ax=ax)
                st.pyplot(fig)

                show_wordcloud(df[df["Predicted"] == 1]["Clean_Review"], "☁️ Positive Review WordCloud")
                show_wordcloud(df[df["Predicted"] == 0]["Clean_Review"], "☁️ Negative Review WordCloud")

                st.download_button("📥 Download Results", data=df.to_csv(index=False), file_name="scraped_reviews_sentiment.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
