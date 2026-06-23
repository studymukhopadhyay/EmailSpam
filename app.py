#---------------- LIBRARIES ----------------
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# ---------------- LOAD DATA ----------------
def load_data():
    df = pd.read_csv("mail_data.csv")
    df = df.where((pd.notnull(df)), '')
    df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})
    return df

df = load_data()

#print(df)

# ---------------- TRAIN MODEL ----------------
def train_model(data):
    X = data['Message']
    y = data['Category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=3
    )

    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_features, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train_features))
    test_acc = accuracy_score(y_test, model.predict(X_test_features))

    return model, vectorizer, train_acc, test_acc

model, vectorizer, train_acc, test_acc = train_model(df)

# ---------------- PREDICTION ----------------
@app.route('/predict', methods=['GET'])
def predict():
    user_input ="Enter a string "
    result = ""
    if user_input.strip() != "":
        input_data = vectorizer.transform([user_input])
        prediction = model.predict(input_data)[0]

        if prediction == 0:
            result= "This message is SPAM"
        else:
            result= "This message is NOT Spam"
    else:
        result = "Please enter a message"
    return result
# ==========================================
# Run Application
# ==========================================
if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )