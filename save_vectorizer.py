"""
save_vectorizer.py
──────────────────
Bu scripti, notebook'unuzu çalıştırdıktan sonra CountVectorizer'ı pickle
formatında kaydetmek için kullanın.

KULLANIM
--------
Seçenek 1 — Notebook'tan dışa aktarma:
    Notebook'unuzdaki "vect" değişkeni fit edildikten sonra aşağıdaki
    hücreyi çalıştırın:

        import pickle
        pickle.dump(vect, open('vectorizer.pkl', 'wb'))
        print("Vectorizer kaydedildi: vectorizer.pkl")

Seçenek 2 — Bu scripti bağımsız çalıştırma:
    Notebook'unuzdan veri seti yolunu güncelleyin ve scripti çalıştırın:

        python save_vectorizer.py

Ardından oluşan "vectorizer.pkl" dosyasını app.py ile aynı dizine koyun.
"""

import pickle
import re
import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import nltk

# NLTK veri setlerini indir (ilk çalıştırmada gerekli)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

# ── Ön İşleme Fonksiyonları ────────────────────────────────────

stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """Notebook ile birebir aynı metin temizleme adımları."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.replace('\n', '').replace('\r', '')
    return text


def ekkok(text):
    """Notebook ile birebir aynı analyzer fonksiyonu (lemmatization + stopword removal)."""
    words = TextBlob(text).words
    return [word.lemmatize() for word in words if word.lower() not in stop_words]


# ── Veri Yükleme ve Vectorizer Eğitimi ────────────────────────

if __name__ == '__main__':
    # Veri setinizin yolunu güncelleyin
    DATA_PATH = 'news.csv'  # ya da 'fake_or_real_news.csv'

    print(f"Veri seti yükleniyor: {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"HATA: '{DATA_PATH}' bulunamadı.")
        print("Lütfen DATA_PATH değişkenini doğru veri seti yoluyla güncelleyin.")
        raise

    # title + text birleştirme (notebook ile aynı)
    df['text'] = df['title'] + " " + df['text']

    # Metin temizleme
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)
    df['text'] = df['text'].str.replace(r'\d+', '', regex=True)
    df['text'] = df['text'].str.replace('\n', '').str.replace('\r', '')

    # CountVectorizer — notebook ile birebir aynı parametreler
    print("CountVectorizer eğitiliyor (bu birkaç dakika sürebilir)...")
    vect = CountVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        min_df=5,
        analyzer=ekkok,
        stop_words='english'
    )
    vect.fit(df['text'])

    # Pickle olarak kaydet
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vect, f)

    print("✅ Vectorizer başarıyla kaydedildi: vectorizer.pkl")
    print(f"   Vocabulary boyutu: {len(vect.vocabulary_)}")
