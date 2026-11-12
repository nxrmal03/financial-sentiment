import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

STOP_WORDS = set(stopwords.words('english'))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return ' '.join(tokens)


def load_data(filepath):
    df = pd.read_csv(filepath, encoding='latin-1', header=None, names=['sentiment', 'sentence'])
    df = df[df['sentiment'].isin(['positive', 'negative', 'neutral'])]
    df['clean'] = df['sentence'].apply(clean_text)
    return df


if __name__ == '__main__':
    df = load_data('all-data.csv')
    print(df['sentiment'].value_counts())
    print(df.head())
