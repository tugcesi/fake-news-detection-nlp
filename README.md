# 📰 Fake News Detection — The Truth Gazette

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00?logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<p align="center">
  <strong>🇹🇷 Türkçe</strong> · <a href="#english">🇬🇧 English</a>
</p>

---

## 🇹🇷 Türkçe

### Proje Hakkında

**The Truth Gazette**, doğal dil işleme (NLP) ve derin öğrenme tekniklerini kullanarak haber metinlerini **Gerçek** veya **Sahte** olarak sınıflandıran bir web uygulamasıdır. Gazete temalı profesyonel bir arayüzle sunulan uygulama, Streamlit ile geliştirilmiştir.

### 🖥️ Uygulama Arayüzü

- **Başlık**: "📰 The Truth Gazette" — Eski tip gazete fontu ile
- **Sidebar**: ASCII art gazete figürleri, model bilgisi, nasıl çalışır açıklaması
- **Ana Alan**: Metin girişi, analiz butonu, renkli sonuç kutusu ve güven çubuğu
- **Tema**: Serif fontlar (UnifrakturMaguntia, Playfair Display, Merriweather), gazete stili kenarlıklar

### ⚙️ Kurulum

```bash
# 1. Repoyu klonlayın
git clone https://github.com/tugcesi/fake-news-detection-nlp.git
cd fake-news-detection-nlp

# 2. Bağımlılıkları kurun
pip install -r requirements.txt

# 3. NLTK veri setlerini indirin
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### 📁 Gerekli Dosyalar

Uygulamayı çalıştırmadan önce şu dosyaların proje dizininde bulunması gerekir:

| Dosya | Açıklama | Nasıl Elde Edilir? |
|-------|----------|--------------------|
| `fakeorreal_model.h5` | Eğitilmiş Keras modeli | `FakeorRealNews.ipynb` çalıştırılarak |
| `vectorizer.pkl` | Eğitilmiş CountVectorizer | `save_vectorizer.py` çalıştırılarak |

#### Vectorizer'ı Kaydetme

```bash
# Veri setini hazırlayın ve save_vectorizer.py içindeki DATA_PATH'i güncelleyin
python save_vectorizer.py
```

Alternatif olarak, notebook'un sonuna şu hücreyi ekleyin:

```python
import pickle
pickle.dump(vect, open('vectorizer.pkl', 'wb'))
```

### 🚀 Çalıştırma

```bash
streamlit run app.py
```

Tarayıcınızda `http://localhost:8501` adresine gidin.

### 🧠 Model Detayları

| Özellik | Detay |
|---------|-------|
| **Mimari** | Keras Sequential |
| **Katmanlar** | Dense(128, relu) → Dropout(0.5) → Dense(64, relu) → Dense(1, sigmoid) |
| **Vectorizer** | CountVectorizer — ngram (1,2), 20.000 özellik, min_df=5 |
| **Analyzer** | TextBlob lemmatization + NLTK stopword removal |
| **Eğitim Doğruluğu** | ~%99.6 |
| **Doğrulama Doğruluğu** | ~%92.3 |
| **Etiketler** | FAKE=0, REAL=1 (sigmoid > 0.5 → REAL) |

### 🛠️ Teknoloji Stack'i

- **Python** 3.9+
- **Streamlit** — Web arayüzü
- **TensorFlow / Keras** — Model yükleme ve tahmin
- **scikit-learn** — CountVectorizer
- **TextBlob** — Lemmatization
- **NLTK** — Stopword listesi
- **NumPy / Pandas** — Veri işleme

---

<a name="english"></a>

## 🇬🇧 English

### About

**The Truth Gazette** is a web application that classifies news articles as **Real** or **Fake** using natural language processing and deep learning. Built with Streamlit, it features a professional newspaper-themed interface.

### ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/tugcesi/fake-news-detection-nlp.git
cd fake-news-detection-nlp

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK datasets
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### 📁 Required Files

Before running the app, make sure these files are in the project directory:

| File | Description | How to get it |
|------|-------------|---------------|
| `fakeorreal_model.h5` | Trained Keras model | Run `FakeorRealNews.ipynb` |
| `vectorizer.pkl` | Fitted CountVectorizer | Run `save_vectorizer.py` |

### 🚀 Running

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

### 🧠 Model Details

| Property | Details |
|----------|---------|
| **Architecture** | Keras Sequential |
| **Layers** | Dense(128, relu) → Dropout(0.5) → Dense(64, relu) → Dense(1, sigmoid) |
| **Vectorizer** | CountVectorizer — ngram (1,2), 20,000 features, min_df=5 |
| **Analyzer** | TextBlob lemmatization + NLTK stopword removal |
| **Training Accuracy** | ~99.6% |
| **Validation Accuracy** | ~92.3% |
| **Labels** | FAKE=0, REAL=1 (sigmoid > 0.5 → REAL) |

### 🛠️ Tech Stack

- **Python** 3.9+
- **Streamlit** — Web interface
- **TensorFlow / Keras** — Model inference
- **scikit-learn** — CountVectorizer
- **TextBlob** — Lemmatization
- **NLTK** — Stopword list
- **NumPy / Pandas** — Data processing

---

📰 *The Truth Gazette © 2026 | Developed with ❤️ by tugcesi*
