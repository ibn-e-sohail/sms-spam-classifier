import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from src.preprocessing import preprocess_dataframe

def train_model():
    print("Loading data...")
    # Load data (adjust path if necessary, assuming we run from root directory)
    df = pd.read_csv('data/spam.csv', encoding='latin-1')
    
    # Drop unnamed columns and rename
    cols_to_drop = [col for col in df.columns if col.startswith('Unnamed')]
    df = df.drop(cols_to_drop, axis=1)
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})

    print("Preprocessing text...")
    df = preprocess_dataframe(df)

    X = df['cleaned_message']
    y = df['label']

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training model...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Evaluate quickly
    accuracy = model.score(X_test_vec, y_test)
    print(f"Model training complete! Test Accuracy: {accuracy:.4f}")

    # Save the vectorizer and model so we can use them later in the API
    print("Saving model and vectorizer...")
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    joblib.dump(model, 'models/spam_classifier.pkl')
    print("Successfully saved to models/ folder!")

if __name__ == "__main__":
    train_model()