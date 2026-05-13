import pandas as pd
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# ---------------- LOAD DATASET ---------------- #

df = pd.read_csv("IMDB Dataset.csv")

# ---------------- NLP SETUP ---------------- #

stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

# ---------------- PREPROCESS FUNCTION ---------------- #

def preprocess(text):

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Keep only letters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Lowercase
    text = text.lower()

    # Split words
    words = text.split()

    # Remove stopwords
    words = [
        word for word in words
        if word not in stop_words
    ]

    # Stemming
    words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(words)

# ---------------- PREPROCESS DATA ---------------- #

df['review'] = df['review'].apply(preprocess)

# ---------------- TRAIN TEST SPLIT ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    df['review'],
    df['sentiment'],
    test_size=0.2,
    random_state=42
)

# ---------------- TF-IDF ---------------- #

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)

# ---------------- MODEL ---------------- #

model = LinearSVC()

# Train
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# ---------------- SAVE FILES ---------------- #

pickle.dump(
    model,
    open("model.pkl", "wb")
)

pickle.dump(
    vectorizer,
    open("vectorizer.pkl", "wb")
)

print("Training completed successfully!")