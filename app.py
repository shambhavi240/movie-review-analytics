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

# ---------------- GLASSMORPHISM UI ---------------- #

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}

/* Text input */
.stTextInput input {
    background-color: rgba(255,255,255,0.12) !important;
    color: white !important;
    border: 2px solid #ff4b4b !important;
    border-radius: 12px !important;
    font-size: 18px !important;
    padding: 10px !important;
}

/* Text area */
.stTextArea textarea {
    background-color: rgba(255,255,255,0.12) !important;
    color: white !important;
    border: 2px solid #ff4b4b !important;
    border-radius: 12px !important;
    font-size: 18px !important;
    padding: 10px !important;
}

/* Dropdown/selectbox */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.12) !important;
    color: white !important;
    border-radius: 12px !important;
    border: 2px solid #ff4b4b !important;
}

/* Dropdown text */
.stSelectbox * {
    color: white !important;
}

/* Labels */
label, .stMarkdown, p, h1, h2, h3 {
    color: white !important;
}

/* Button */
.stButton button {
    width: 100%;
    height: 50px;
    border-radius: 15px;
    font-size: 18px;
    font-weight: bold;
    background: linear-gradient(to right, #ff512f, #dd2476);
    color: white;
    border: none;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: rgba(255,255,255,0.08);
    border-radius: 10px;
}

/* Metrics */
[data-testid="metric-container"] {
    background-color: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

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

        clean_review = preprocess(review)

        review_vec = vectorizer.transform([clean_review])

        prediction = model.predict(review_vec)[0]

        probability = model.predict_proba(review_vec)[0]

        if prediction == 1:

            sentiment = "Positive"

            confidence = probability[1] * 100

            st.success(
                f"✅ Positive Review ({confidence:.2f}% confidence)"
            )

            st.balloons()

        else:

            sentiment = "Negative"

            confidence = probability[0] * 100

            st.error(
                f"❌ Negative Review ({confidence:.2f}% confidence)"
            )

        # SAVE REVIEW
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
            confidence
        ))

        conn.commit()

        st.success("💾 Review Saved!")

    else:
        st.warning("Please enter movie name and review.")

# ---------------- ANALYTICS ---------------- #

st.markdown("---")

st.header("📊 Movie Analytics")

movies_df = pd.read_sql_query(
    "SELECT * FROM reviews",
    conn
)

if not movies_df.empty:

    movie_list = movies_df['movie_name'].unique()

    selected_movie = st.selectbox(
        "Select Movie",
        movie_list
    )

    movie_reviews = movies_df[
        movies_df['movie_name'] == selected_movie
    ]

    # FILTER REVIEWS
    filter_option = st.selectbox(
        "Filter Reviews",
        ['All', 'Positive', 'Negative']
    )

    if filter_option != 'All':
        movie_reviews = movie_reviews[
            movie_reviews['sentiment'] == filter_option
        ]

    total_reviews = len(movie_reviews)

    positive_reviews = len(
        movie_reviews[
            movie_reviews['sentiment'] == 'Positive'
        ]
    )

    negative_reviews = len(
        movie_reviews[
            movie_reviews['sentiment'] == 'Negative'
        ]
    )

    positive_percent = (
        positive_reviews / total_reviews
    ) * 100 if total_reviews > 0 else 0

    negative_percent = (
        negative_reviews / total_reviews
    ) * 100 if total_reviews > 0 else 0

    # POSTER
    if selected_movie in movie_posters:
        st.image(
            movie_posters[selected_movie],
            width=250
        )

    # METRICS
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Reviews",
        total_reviews
    )

    col2.metric(
        "👍 Positive %",
        f"{positive_percent:.1f}%"
    )

    col3.metric(
        "👎 Negative %",
        f"{negative_percent:.1f}%"
    )

    # AVERAGE SCORE
    avg_score = positive_percent - negative_percent

    st.metric(
        "⭐ Sentiment Score",
        f"{avg_score:.1f}"
    )

    # PIE CHART
    chart_df = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative'],
        'Count': [positive_reviews, negative_reviews]
    })

    fig = px.pie(
        chart_df,
        names='Sentiment',
        values='Count',
        hole=0.4,
        title='Sentiment Distribution'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

        # REVIEW HISTORY
    st.subheader("📝 Review History")

    columns_to_show = ['review', 'sentiment']

    if 'confidence' in movie_reviews.columns:
        columns_to_show.append('confidence')

    if 'timestamp' in movie_reviews.columns:
        columns_to_show.append('timestamp')

    st.dataframe(
        movie_reviews[columns_to_show]
    )

    # TOP RATED MOVIES
    st.markdown("---")

    st.header("🏆 Top Rated Movies")

    movie_stats = movies_df.groupby(
        'movie_name'
    )['sentiment'].apply(
        lambda x: (x == 'Positive').mean() * 100
    ).reset_index(name='positive_percent')

    movie_stats = movie_stats.sort_values(
        by='positive_percent',
        ascending=False
    )

    st.dataframe(movie_stats)

    # MOST REVIEWED MOVIES
    st.markdown("---")

    st.header("🔥 Most Reviewed Movies")

    most_reviewed = movies_df[
        'movie_name'
    ].value_counts().reset_index()

    most_reviewed.columns = [
        'Movie',
        'Reviews'
    ]

    st.dataframe(most_reviewed)

    # TRENDING MOVIES
    st.markdown("---")

    st.header("📈 Trending Movies")

    trending = movies_df.groupby(
        'movie_name'
    ).size().reset_index(name='count')

    trend_fig = px.bar(
        trending,
        x='movie_name',
        y='count',
        title='Trending Movies by Reviews'
    )

    st.plotly_chart(
        trend_fig,
        use_container_width=True
    )

else:
    st.info("No reviews available yet.")