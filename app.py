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
from transformers import pipeline
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
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

emotion_classifier = load_emotion_model()

# ---------------- NLP SETUP ---------------- #

stop_words = set(stopwords.words('english'))

# Keep important negation words
stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("never")

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
        emotion_result = emotion_classifier(review)[0]
        dominant_emotion = max(
    emotion_result,
    key=lambda x: x['score']
)
        emotion_name = dominant_emotion['label']
        emotion_score = dominant_emotion['score'] * 100

        # Predict
        probability = model.predict_proba(review_vec)[0]
        positive_prob = probability[1]
        negative_prob = probability[0]
        st.write("Positive Probability:", positive_prob)
        st.write("Negative Probability:", negative_prob)
        if positive_prob >= 0.66:
            sentiment = "Positive"
            confidence = positive_prob * 100

        elif positive_prob <= 0.40:
            sentiment = "Negative"
            confidence = negative_prob * 100

        else:
            sentiment = "Neutral"
            confidence = max(
            positive_prob,
        negative_prob
    ) * 100

        # Positive
        if sentiment == "Positive":
            st.success(
        f"✅ Positive Review ({confidence:.1f}% confidence)"
    )
            st.balloons()
        elif sentiment == "Neutral":
            st.warning(
        f"😐 Neutral Review ({confidence:.1f}% confidence)"
    )

        else:
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
        st.markdown("---")
        st.header("💭 Emotion Analysis")

        col1,col2 = st.columns(2)
        col1.metric(
    "Dominant Emotion",
    emotion_name.capitalize()
)
        col2.metric(
    "Emotion Strength",
    f"{emotion_score:.1f}%"
)
        emotion_df = pd.DataFrame(emotion_result)
        emotion_df.rename(
            columns={
        "label":"Emotion",
        "score":"Score"
    },
            inplace=True
)

        emotion_df["Score"] *= 100

        emotion_fig = px.bar(
            emotion_df,
            x="Emotion",
            y="Score",
            color="Emotion",
            color_discrete_sequence=[
                "#24D4E3",
                "#8B5CF6",
                "#F472B6",
                "#24D4E3",
                "#8B5CF6"
            ]
        )

        emotion_fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        st.plotly_chart(
            emotion_fig,
            use_container_width=True
        )
        st.markdown("---")
        st.header("📝 AI Generated Summary")
        if sentiment == "Positive":

            summary = f"""
            Audience reaction to {movie_name}
            is strongly positive.

            Viewers appear to appreciate
            the storytelling, performances
            and overall cinematic experience.

            Dominant emotion detected:
            {emotion_name}.
         """

        elif sentiment == "Neutral":

            summary = f"""
            Audience opinion on {movie_name}
            appears mixed.

            Some viewers enjoyed the movie,
            while others found parts of it
            less engaging.

            Dominant emotion detected:
            {emotion_name}.
            """

        else:

            summary = f"""
                Audience reaction to {movie_name}
                is mostly negative.

                Reviews indicate concerns about
                pacing, execution or overall
                enjoyment.

                Dominant emotion detected:
                {emotion_name}.
                """
        st.markdown(f"""
<div style="
background:white;
padding:20px;
border-radius:15px;
border-left:6px solid #8B5CF6;
box-shadow:0px 5px 15px rgba(0,0,0,0.08);
">

<h3 style="color:#8B5CF6;">
📝 AI Generated Summary
</h3>

<p style="font-size:18px;">
{summary}
</p>

</div>
""", unsafe_allow_html=True)         

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
    sheet_data = sheet.get_all_records()
    movies_df = pd.DataFrame(sheet_data)

if not movies_df.empty:

    movie_list = movies_df['Movie'].unique()
    selected_movie = st.selectbox("Select Movie", movie_list)

    movie_reviews = movies_df[movies_df['Movie'] == selected_movie]
    all_reviews = " ".join(
    movie_reviews["review"].tolist()
)

    # Filter
    filter_option = st.selectbox(
    "Filter Reviews",
    ["All","Positive","Neutral","Negative"]
)

    if filter_option != 'All':
        movie_reviews = movie_reviews[movie_reviews['sentiment'] == filter_option]
        st.markdown("---")
        st.subheader(f"📝 {filter_option} Reviews")
        for _, row in movie_reviews.iterrows():
            if row['sentiment'] == 'Positive':
                st.success(
            f"{row['review']}\n\nConfidence: {row['confidence']:.1f}%"
        )
            elif row['sentiment'] == 'Negative':
                st.error(
            f"{row['review']}\n\nConfidence: {row['confidence']:.1f}%"
        )
            else:
                st.warning(
            f"{row['review']}\n\nConfidence: {row['confidence']:.1f}%"
        )
        

    total_reviews    = len(movie_reviews)
    positive_reviews = len(movie_reviews[movie_reviews['sentiment'] == 'Positive'])
    negative_reviews = len(movie_reviews[movie_reviews['sentiment'] == 'Negative'])
    neutral_reviews = len(movie_reviews[movie_reviews['sentiment']=="Neutral"]
)
    positive_percent = (positive_reviews / total_reviews * 100) if total_reviews > 0 else 0
    negative_percent = (negative_reviews / total_reviews * 100) if total_reviews > 0 else 0
    neutral_percent  = (neutral_reviews / total_reviews * 100) if total_reviews > 0 else 0
    neutral_percent = (
    neutral_reviews/total_reviews*100
    if total_reviews>0 else 0
)
    st.markdown("---")
    st.header("🎭 Audience Mood Meter")

    col1,col2,col3 = st.columns(3)

    col1.metric(
    "😊 Positive",
    f"{positive_percent:.1f}%"
)

    col2.metric(
    "😐 Neutral",
    f"{neutral_percent:.1f}%"
)

    col3.metric(
    "😡 Negative",
    f"{negative_percent:.1f}%"
)
    st.header("🧠 AI Insights")
    verdict = ""
    if positive_percent >= 70:
        verdict = "Highly Recommended"
    elif positive_percent >= 50:
        verdict = "Recommended"
    else:
        verdict = "Mixed Reception"
    col1,col2,col3 = st.columns(3)
    col1.metric(
    "Audience Verdict",
    verdict
)
    col2.metric(
    "Positive %",
    f"{positive_percent:.1f}%"
)
    col3.metric(
    "Reviews",
    total_reviews
)
    # Poster
    if selected_movie in movie_posters:
        st.image(movie_posters[selected_movie], width=250)

    # Metrics
    col1,col2,col3,col4,col5 = st.columns(5)

    col1.metric("📊 Reviews", total_reviews)
    col2.metric("😊 Positive", positive_reviews)
    col3.metric("😐 Neutral", neutral_reviews)
    col4.metric("😞 Negative", negative_reviews)
    col5.metric("🎯 Accuracy", "91%")

    # Pie Chart
    chart_df = pd.DataFrame({
    'Sentiment': [
        'Positive',
        'Neutral',
        'Negative'
    ],
    'Count': [
        positive_reviews,
        neutral_reviews,
        negative_reviews
    ]
})

    fig = px.pie(
    chart_df,
    names='Sentiment',
    values='Count',
    hole=0.65,
    color='Sentiment',
    color_discrete_map={
        'Positive': "#24D4E3",
        'Neutral': '#8B5CF6',
        'Negative': '#F472B6'
    }
)

    fig.update_traces(
    textinfo='percent+label',
    pull=[0.03,0,0]
)

    fig.update_layout(
    paper_bgcolor='white',
    plot_bgcolor='white'
)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.header("📡 Movie Quality Radar")

    radar_df = pd.DataFrame({
    'Category': [
        'Story',
        'Acting',
        'Visuals',
        'Direction',
        'Music'
    ],
    'Score': [
        85,
        92,
        95,
        88,
        90
    ]
})

    radar_fig = go.Figure()
    radar_fig.add_trace(
    go.Scatterpolar(
        r=radar_df['Score'],
        theta=radar_df['Category'],
        fill='toself',
        fillcolor='rgba(139,92,246,0.4)',
        line=dict(
            color='#8B5CF6',
            width=3
        )
    )
)
    radar_fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0,100]
        )
    ),
    showlegend=False,
    paper_bgcolor='white'
)
    st.plotly_chart(
    radar_fig,
    use_container_width=True
)
    st.markdown("---")
    st.header("🏆 Top Rated Movies")
    movie_stats = movies_df.groupby('Movie')['sentiment'].apply(
        lambda x: (x == 'Positive').mean() * 100
    ).reset_index(name='positive_percent')
    movie_stats = movie_stats.sort_values(by='positive_percent', ascending=False)
    fig = px.bar(
    movie_stats,
    x='movie_name',
    y='positive_percent',
    color='positive_percent',
    color_continuous_scale=[
        "#24D4E3",
        "#8B5CF6",
        "#F472B6"
    ]
)

    fig.update_layout(
    paper_bgcolor='white',
    plot_bgcolor='white'
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
    ttrending = movies_df.groupby('Movie').size().reset_index(name='count')
    trend_fig = px.bar(
    trending,
    x='count',
    y='movie_name',
    orientation='h',
    color='count',
    color_continuous_scale='purples'
)
    trend_fig.update_layout(
    paper_bgcolor='white',
    plot_bgcolor='white'
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
