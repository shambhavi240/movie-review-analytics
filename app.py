import streamlit as st
import pickle
import re
import sqlite3
import pandas as pd
import nltk
import plotly.express as px
import plotly.graph_objects as go
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
# ---------------- GOOGLE SHEETS ---------------- #

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    dict(st.secrets),
    scope
)

client = gspread.authorize(creds)

sheet = client.open(
    "Movie Reviews Database"
).sheet1
# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Movie Review Analytics",
    page_icon="🎬",
    layout="wide"
)
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F5F7FB;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #24D4E3,
        #8B5CF6
    );
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Headings */
h1 {
    color: #334155 !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #475569 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #64748B;
}

.stTabs [aria-selected="true"] {
    color: #8B5CF6 !important;
}

</style>
""", unsafe_allow_html=True)

import os
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

cursor.execute("SELECT COUNT(*) FROM reviews")
count = cursor.fetchone()[0]
with st.sidebar:



    st.title("🎬 Dashboard")



    st.markdown("---")



    st.subheader("🤖 Model")

    st.write("Logistic Regression")

    st.write("TF-IDF Vectorizer")



    st.markdown("---")



    st.subheader("📈 Statistics")

    st.metric(

    "Total Reviews",

    count

)
    st.markdown("---")
    st.info(

        "AI Powered Movie Sentiment Analysis System"

    )

# ---------------- LOAD MODEL ---------------- #

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- NLP SETUP ---------------- #

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# ---------------- PREPROCESS FUNCTION ---------------- #

def preprocess(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
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

st.markdown("""
<h1 style="
text-align:center;
color:#0F172A;
font-size:55px;
font-weight:800;
">
🎬 Movie Review Analytics
</h1>
""", unsafe_allow_html=True)
st.markdown("""
<div style="
background: linear-gradient(
90deg,
#24D4E3,
#8B5CF6
);
padding:20px;
border-radius:18px;
text-align:center;
font-size:26px;
font-weight:700;
color:white;
box-shadow:0px 8px 20px rgba(0,0,0,0.08);
">
🎯 AI Powered Sentiment Intelligence Platform
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- INPUTS ---------------- #

movie_name = st.text_input("🎥 Enter Movie Name")
review = st.text_area("✍️ Enter Review", height=200)

# ---------------- BUTTON ---------------- #

if st.button("Analyze & Save Review"):

    if movie_name.strip() != "" and review.strip() != "":

        # Preprocess review
        clean_review = preprocess(review)

        # Vectorize
        review_vec = vectorizer.transform([clean_review])

        # Predict
        prediction = model.predict(review_vec)[0]

        # Probability
        probability = model.predict_proba(review_vec)[0]

        # Positive
        if prediction == "positive":

            sentiment = "Positive"
            confidence = probability[1] * 100

            st.success(
                f"✅ Positive Review ({confidence:.1f}% confidence)"
            )

            st.balloons()

        # Negative
        else:

            sentiment = "Negative"
            confidence = probability[0] * 100

            st.error(
                f"❌ Negative Review ({confidence:.1f}% confidence)"
            )

        # Metrics
        col1,col2,col3 = st.columns(3)
        with col1:
            st.metric("😊 Sentiment", sentiment)
        with col2:
            st.metric("🎯 Confidence", f"{confidence:.2f}%")
        with col3:
            st.metric("🎬 Movie", movie_name)

        st.progress(confidence / 100)

        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={"text": "Confidence Score"},
                gauge={
                    "axis": {"range": [0, 100]}
                }
            )
        )

        st.plotly_chart(
            gauge_fig,
            use_container_width=True
        )
    

        word_count = len(review.split())

        st.metric(
            "Review Length",
            f"{word_count} words"
        )

        # Google Sheets Save
        from datetime import datetime

        sheet.append_row([
            movie_name,
            review,
            sentiment,
            f"{confidence:.2f}%",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])

        # Save to database
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

        st.success("💾 Review Saved Successfully!")

    else:

        st.warning(
            "⚠️ Please enter movie name and review."
        )
# ---------------- ANALYTICS ---------------- #

tab1, tab2 = st.tabs([
    "📊 Analytics",
    "📜 History"
])

with tab1:

    cursor.execute("""
    SELECT movie_name,
           review,
           sentiment,
           confidence,
           timestamp
    FROM reviews
    ORDER BY id DESC
    """)

rows = cursor.fetchall()

if rows:

    import pandas as pd

    df = pd.DataFrame(
        rows,
        columns=[
            "Movie",
            "Review",
            "Sentiment",
            "Confidence",
            "Timestamp"
        ]
    )

    st.subheader("📜 Reviews")
    st.dataframe(df, use_container_width=True)

else:
    st.info("No reviews available yet.")
movies_df = pd.read_sql_query("SELECT * FROM reviews", conn)

if not movies_df.empty:

    movie_list = movies_df['movie_name'].unique()
    selected_movie = st.selectbox("Select Movie", movie_list)

    movie_reviews = movies_df[movies_df['movie_name'] == selected_movie]

    # Filter
    filter_option = st.selectbox("Filter Reviews", ['All', 'Positive', 'Negative'])

    if filter_option != 'All':
        movie_reviews = movie_reviews[movie_reviews['sentiment'] == filter_option]

    total_reviews    = len(movie_reviews)
    positive_reviews = len(movie_reviews[movie_reviews['sentiment'] == 'Positive'])
    negative_reviews = len(movie_reviews[movie_reviews['sentiment'] == 'Negative'])

    positive_percent = (positive_reviews / total_reviews * 100) if total_reviews > 0 else 0
    negative_percent = (negative_reviews / total_reviews * 100) if total_reviews > 0 else 0

    # Poster
    if selected_movie in movie_posters:
        st.image(movie_posters[selected_movie], width=250)

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", total_reviews)
    col2.metric("👍 Positive %", f"{positive_percent:.1f}%")
    col3.metric("👎 Negative %", f"{negative_percent:.1f}%")

    # Pie Chart
    chart_df = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative'],
        'Count': [positive_reviews, negative_reviews]
    })

    fig = px.pie(
    chart_df,
    names='Sentiment',
    values='Count',
    hole=0.6,
    color='Sentiment',
    color_discrete_map={
        'Positive': '#8B5CF6',
        'Negative': '#F472B6'
    }
)
    st.plotly_chart(fig, use_container_width=True)

    # Top Rated Movies
    st.markdown("---")
    st.header("🏆 Top Rated Movies")

    movie_stats = movies_df.groupby('movie_name')['sentiment'].apply(
        lambda x: (x == 'Positive').mean() * 100
    ).reset_index(name='positive_percent')

    movie_stats = movie_stats.sort_values(by='positive_percent', ascending=False)
    fig = px.bar(
        movie_stats,
        x='movie_name',
        y='positive_percent',
        title='Top Rated Movies'
    )

    st.plotly_chart(
    fig,
    use_container_width=True
)
    
    best_movie = movie_stats.iloc[0]
    st.success(
    f"💡 AI Insight: {best_movie['movie_name']} is currently the highest rated movie with {best_movie['positive_percent']:.1f}% positive reviews."
)

# Trending Movies
    st.markdown("---")
    st.header("📈 Trending Movies")

    trending = movies_df.groupby('movie_name').size().reset_index(name='count')

    trend_fig = px.bar(
    trending,
    x='movie_name',
    y='count',
    title='Trending Movies'
)

    st.plotly_chart(
    trend_fig,
    use_container_width=True
)
else:
    st.info("No reviews available yet.")
st.markdown("---")

st.subheader("🤖 Model Information")

st.write("Algorithm: Logistic Regression")
st.write("Vectorizer: TF-IDF")
st.write("Dataset: IMDb 50K Reviews")
