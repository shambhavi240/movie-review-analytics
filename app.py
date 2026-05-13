import streamlit as st
import pickle
import re
import sqlite3
import pandas as pd
import nltk
import plotly.express as px

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Movie Review Analytics",
    page_icon="🎬",
    layout="wide"
)

# ---------------- DOWNLOAD NLTK ---------------- #

nltk.download('stopwords')

# ---------------- DATABASE ---------------- #

conn = sqlite3.connect("reviews.db", check_same_thread=False)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_name TEXT,
    review TEXT,
    sentiment TEXT,
    confidence REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

# ---------------- LOAD MODEL ---------------- #

model = pickle.load(open("model.pkl", "rb"))

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- NLP SETUP ---------------- #

stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

# ---------------- PREPROCESS FUNCTION ---------------- #

def preprocess(text):

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'[^a-zA-Z\\s]', '', text)

    text = text.lower()

    words = text.split()

    words = [word for word in words if word not in stop_words]

    words = [stemmer.stem(word) for word in words]

    return " ".join(words)

# ---------------- MOVIE POSTERS ---------------- #

movie_posters = {
    "Interstellar": "https://m.media-amazon.com/images/I/71n58R0h5SL._AC_UF894,1000_QL80_.jpg",
    "The Dark Knight": "https://m.media-amazon.com/images/I/51EbJjlYF-L.jpg",
    "Titanic": "https://m.media-amazon.com/images/I/71y6lU9i5yL._AC_UF894,1000_QL80_.jpg",
    "Joker": "https://m.media-amazon.com/images/I/71h0nI4VbUL._AC_UF894,1000_QL80_.jpg",
    "Oppenheimer": "https://m.media-amazon.com/images/I/81J5c7P7QUL._AC_UF894,1000_QL80_.jpg"
}



# ---------------- TITLE ---------------- #

st.title("🎬 Movie Review Analytics System")

st.markdown("""
<h3 style='color: lightgray;'>
AI-powered Movie Sentiment Dashboard
</h3>
""", unsafe_allow_html=True)

# ---------------- INPUTS ---------------- #

movie_name = st.text_input("🎥 Enter Movie Name")

review = st.text_area(
    "✍️ Enter Review",
    height=200
)

# ---------------- BUTTON ---------------- #

if st.button("Analyze & Save Review"):

    if movie_name.strip() != "" and review.strip() != "":

        # Preprocess review
        clean_review = preprocess(review)

        # Convert review into vector
        review_vec = vectorizer.transform([clean_review])

        # Predict sentiment
        prediction = model.predict(review_vec)[0]

        # Positive review
        if prediction == "positive":

            sentiment = "Positive"

            st.success("✅ Positive Review")

            st.balloons()

        # Negative review
        else:

            sentiment = "Negative"

            st.error("❌ Negative Review")

        # Save review into database
        cursor.execute("""
        INSERT INTO reviews (
            movie_name,
            review,
            sentiment,
            confidence
        )
        VALUES (?, ?, ?, ?)
        """, (
            movie_name,
            review,
            sentiment,
            0
        ))

        conn.commit()

        st.success("💾 Review Saved Successfully!")

    else:

        st.warning("⚠️ Please enter movie name and review.")