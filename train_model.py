import pandas as pd
import nltk
import re
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")

# Stopwords + stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Text cleaning function
def preprocess(text):

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Lowercase
    text = text.lower()

    # Split words
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Stemming
    words = [stemmer.stem(word) for word in words]

    return " ".join(words)

# Clean reviews
df['clean_review'] = df['review'].apply(preprocess)

# Features and labels
X = df['clean_review']

y = df['sentiment'].map({
    'positive': 1,
    'negative': 0
})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=10000)

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=200)

# Train model
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")