import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nlp_utils import NLPPreprocessor

# Load dataset
def train_models():
    # Use the sample dataset we created
    data_path = 'data/news_dataset.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    
    # Preprocess text
    preprocessor = NLPPreprocessor()
    df['cleaned_news'] = df['news'].apply(preprocessor.clean_text)

    # Feature extraction (TF-IDF)
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned_news'])
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    # 2. Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)

    # Evaluation metrics
    metrics = {
        'Logistic Regression': {
            'accuracy': accuracy_score(y_test, lr_preds),
            'precision': precision_score(y_test, lr_preds),
            'recall': recall_score(y_test, lr_preds),
            'f1': f1_score(y_test, lr_preds)
        },
        'Naive Bayes': {
            'accuracy': accuracy_score(y_test, nb_preds),
            'precision': precision_score(y_test, nb_preds),
            'recall': recall_score(y_test, nb_preds),
            'f1': f1_score(y_test, nb_preds)
        }
    }

    print("Model Evaluation Metrics:")
    for model_name, m in metrics.items():
        print(f"\n{model_name}:")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")

    # Save models and vectorizer
    os.makedirs('models', exist_ok=True)
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    with open('models/lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    with open('models/nb_model.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    
    print("\nModels and vectorizer saved to 'models/'")

if __name__ == "__main__":
    train_models()
