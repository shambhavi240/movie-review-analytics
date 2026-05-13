from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")

# Convert labels
df['sentiment'] = df['sentiment'].map({
    'positive': 'positive',
    'negative': 'negative'
})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['review'],
    df['sentiment'],
    test_size=0.2,
    random_state=42
)

# Vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)

# Better model
model = LinearSVC()

# Train
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save files
pickle.dump(
    model,
    open("model.pkl", "wb")
)

pickle.dump(
    vectorizer,
    open("vectorizer.pkl", "wb")
)

print("Model trained successfully!")