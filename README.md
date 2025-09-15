import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = {
    "message": [
        "Congratulations! You won a lottery of $1,000,000",
        "Hello friend, how are you doing today?",
        "Free entry in 2 crore prize, click the link now",
        "Let’s catch up tomorrow for lunch",
        "Win cash instantly!!! Claim your prize",
        "Reminder: Your appointment is scheduled tomorrow",
        "You have been selected for a free gift voucher",
        "Meeting is rescheduled to 4 PM today"
    ],
    "label": ["spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham"]
}

df = pd.DataFrame(data)


df["label"] = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.25, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print(" Accuracy:", accuracy_score(y_test, y_pred))


test_messages = [
    "Congratulations! You won a free iPhone!",
    "Hey, are we meeting for lunch tomorrow?",
    "Get premium movies free now!"
]

test_vec = vectorizer.transform(test_messages)
predictions = model.predict(test_vec)

for msg, pred in zip(test_messages, predictions):
    print(f"Message: {msg}\nPrediction: {'Spam' if pred == 1 else 'Ham'}\n")
