import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from preprocess import clean_text


def load_best_data():
    # Use 75% agreement file for cleaner labels + all-data for extra volume
    rows = []
    try:
        with open('FinancialPhraseBank/Sentences_75Agree.txt', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if '@' in line:
                    parts = line.rsplit('@', 1)
                    if len(parts) == 2:
                        sentence, label = parts[0].strip(), parts[1].strip()
                        if label in ('positive', 'negative', 'neutral'):
                            rows.append({'sentence': sentence, 'sentiment': label})
    except FileNotFoundError:
        pass

    # Fallback to all-data.csv if 75Agree not found
    if not rows:
        df = pd.read_csv('all-data.csv', encoding='latin-1', header=None,
                         names=['sentiment', 'sentence'])
        df = df[df['sentiment'].isin(['positive', 'negative', 'neutral'])]
        return df

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} rows from Sentences_75Agree.txt")
    print(df['sentiment'].value_counts())
    return df


def train():
    df = load_best_data()
    df['clean'] = df['sentence'].apply(clean_text)

    X = df['clean']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Improved TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # LinearSVC wrapped with calibration for probability scores
    base_model = LinearSVC(C=1.0, max_iter=2000, class_weight='balanced')
    model = CalibratedClassifierCV(base_model, cv=5)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(classification_report(y_test, y_pred))

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("Model saved to model.pkl and vectorizer.pkl")


if __name__ == '__main__':
    train()
