# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

def load_data(path: str = "data/comments_labeled.csv"):
    df = pd.read_csv(path, encoding='utf-8')
    df = df.dropna(subset=["text", "label"])
    print("Label distribution:\n", df['label'].value_counts())
    return df

def train_logistic(df: pd.DataFrame):
    """Your original Logistic Regression model"""
    X = df["text"]
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Train label counts:\n", y_train.value_counts())
    print("Val label counts:\n", y_val.value_counts())

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            #multi_class="auto"
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    print("\n=== Logistic Regression Results ===\n")
    print(classification_report(y_val, y_pred))
    
    return model

def train_svm(df: pd.DataFrame):
    """NEW: LinearSVC model"""
    X = df["text"]
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2,
            stop_words="english"
        )),
        ("clf", LinearSVC(
            class_weight="balanced",
            max_iter=2000,
            C=0.1
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    print("\n=== LinearSVC (SVM) Results ===\n")
    print(classification_report(y_val, y_pred))
    
    return model

def main():
    df = load_data()
        # Train BOTH models
    print("Training Logistic Regression...")
    lr_model = train_logistic(df)
    joblib.dump(lr_model, "models/logistic_classifier.pkl")
    print("Logistic model saved to models/logistic_classifier.pkl")
    
    print("\nTraining LinearSVC (SVM)...")
    svm_model = train_svm(df)
    joblib.dump(svm_model, "models/svm_classifier.pkl")
    print("SVM model saved to models/svm_classifier.pkl")
    
    print("\n✅ Both models trained and saved!")

if __name__ == "__main__":
    main()
