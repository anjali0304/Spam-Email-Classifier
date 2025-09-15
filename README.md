Spam Email Classifier

Spam Email Classifier is a Python-based machine learning project that automatically detects whether an email is Spam or Ham. The system uses TF-IDF vectorization to convert text into numerical features and Multinomial Naive Bayes for classification. It demonstrates the full workflow of text preprocessing, feature extraction, model training, evaluation, and prediction.

Dataset
->The project uses email messages labeled as Spam or Ham.
 ->Each record contains:
message: The text of the email.
label: The class (spam or ham).
->You can use a small sample dataset included in the repository or a public dataset like the SMS Spam Collection Dataset from Kaggle.
->All datasets are simulated or publicly available, with no personal or sensitive information used.
->Users can also test with custom email messages for predictions.

Code Example:

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = {
    "message": [
        "Congratulations! You won a lottery",
        "Hello friend, how are you?",
        "Win cash instantly! Claim your prize",
        "Meeting is rescheduled to 4 PM"
    ],
    "label": ["spam", "ham", "spam", "ham"]
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
print("Accuracy:", accuracy_score(y_test, y_pred))

Explanation:
->Data Preparation: A small dataset of email messages labeled as spam or ham is used. Labels are converted to 0 (ham) and 1 (spam).
->TF-IDF Vectorization: Converts text into numerical features so the model can process it.
->Model Training: A Multinomial Naive Bayes model is trained on the vectorized data.
->Evaluation: Accuracy is calculated to check model performance.
->Prediction: The trained model can classify new email messages as Spam or Ham.

Setup & Usage
Clone the repository:
git clone <your-repo-link>
cd spam-email-classifier

Install dependencies:
pip install -r requirements.txt

Run the classifier:
python spam_classifier.py

Test with custom messages included in the script.

Technologies:
Python
pandas
scikit-learn
TF-IDF Vectorization
Multinomial Naive Bayes
