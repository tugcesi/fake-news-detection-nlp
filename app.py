import streamlit as st
import numpy as np
import re
import pickle
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
from textblob import TextBlob
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def ekkok(text):
    words = TextBlob(text).words
    return [word.lemmatize() for word in words if word.lower() not in stop_words]

# ── Sayfa Ayarları ──────────────────────────────────────────────
st.set_page_config(
    page_title="📰 The Truth Gazette",
    page_icon="🗞️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── CSS ile Gazete Teması ───────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400&family=Merriweather:ital,wght@0,400;1,400&family=UnifrakturMaguntia&display=swap');

    .main-header {
        text-align: center;
        padding: 20px 0 10px 0;
        border-bottom: 4px double #222;
        margin-bottom: 5px;
    }
    .main-header h1 {
        font-family: 'UnifrakturMaguntia', cursive;
        font-size: 3.2rem;
        color: #1a1a1a;
        margin: 0;
        letter-spacing: 2px;
    }
    .sub-header {
        text-align: center;
        font-family: 'Playfair Display', serif;
        font-size: 0.95rem;
        color: #555;
        border-bottom: 2px solid #222;
        padding-bottom: 12px;
        margin-bottom: 25px;
    }
    .newspaper-divider {
        border: none;
        border-top: 1px solid #999;
        margin: 20px 0;
    }
    .result-box-fake {
        background: linear-gradient(135deg, #ff4b4b22, #ff000011);
        border-left: 6px solid #ff4b4b;
        padding: 20px 25px;
        border-radius: 0 10px 10px 0;
        margin: 15px 0;
        font-family: 'Merriweather', serif;
    }
    .result-box-real {
        background: linear-gradient(135deg, #21c35422, #00ff0011);
        border-left: 6px solid #21c354;
        padding: 20px 25px;
        border-radius: 0 10px 10px 0;
        margin: 15px 0;
        font-family: 'Merriweather', serif;
    }
    .confidence-label {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
        font-weight: 700;
    }
    .newspaper-quote {
        font-family: 'Merriweather', serif;
        font-style: italic;
        text-align: center;
        color: #666;
        padding: 10px 30px;
        font-size: 0.9rem;
    }
    .footer-newspaper {
        text-align: center;
        font-family: 'Playfair Display', serif;
        font-size: 0.8rem;
        color: #aaa;
        border-top: 2px solid #222;
        padding-top: 15px;
        margin-top: 40px;
    }
    .stTextArea textarea {
        font-family: 'Merriweather', serif;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Gazete Başlığı ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📰 The Truth Gazette</h1>
</div>
<div class="sub-header">
    Est. 2026 &nbsp;·&nbsp; Yapay Zekâ Destekli Haber Doğrulama Platformu &nbsp;·&nbsp;
    "Gerçeği Yalanlardan Ayırın"
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="newspaper-quote">
    "Bir haberin gerçek olup olmadığını anlamanın en iyi yolu,
    onu sorgulamaktan geçer." — The Truth Gazette
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="newspaper-divider">', unsafe_allow_html=True)


# ── Model ve Vectorizer Yükleme ────────────────────────────────
@st.cache_resource
def load_model():
    from tensorflow.keras.models import load_model as keras_load_model
    try:
        model = keras_load_model("fakeorreal_model.h5")
        return model
    except FileNotFoundError:
        return None


@st.cache_resource
def load_vectorizer():
    try:
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return vectorizer
    except FileNotFoundError:
        return None


# ── Metin Ön İşleme ────────────────────────────────────────────
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.replace('\n', '').replace('\r', '')
    return text


# ── Tahmin Fonksiyonu ──────────────────────────────────────────
def predict_news(text, model, vectorizer):
    cleaned = preprocess_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features, verbose=0)
    confidence = float(prediction[0][0])
    is_real = confidence > 0.5
    real_score = confidence if is_real else 1 - confidence
    fake_score = 1 - confidence if is_real else confidence
    return is_real, real_score, fake_score


# ── Sidebar — Gazete Figürleri & Bilgi ─────────────────────────
with st.sidebar:
    st.markdown("## 🗞️ Gazete Köşesi")
    st.markdown("---")

    st.markdown("""
    ```
    ╔═══════════════════════╗
    ║                       ║
    ║    E X T R A !        ║
    ║    E X T R A !        ║
    ║                       ║
    ║  READ ALL ABOUT IT!   ║
    ║                       ║
    ║  ┌─────┐  SAHTE HABER ║
    ║  │░░░░░│  her yıl     ║
    ║  │░░░░░│  milyarlarca ║
    ║  └─────┘  $ zarar     ║
    ║                       ║
    ║  ▸ Kaynağı doğrula    ║
    ║  ▸ Tarihi kontrol et  ║
    ║  ▸ Yazarı araştır     ║
    ║                       ║
    ╚═══════════════════════╝
    ```
    """)

    st.markdown("---")
    st.markdown("### 📋 Nasıl Çalışır?")
    st.info(
        "1️⃣ Haber metnini metin kutusuna girin\n\n"
        "2️⃣ **HABERİ ANALİZ ET** butonuna tıklayın\n\n"
        "3️⃣ Yapay zekâ modeli metni analiz eder\n\n"
        "4️⃣ Sonuç: **Gerçek** veya **Sahte**"
    )

    st.markdown("---")
    st.markdown("### 📊 Model Bilgisi")
    st.markdown("""
    | Özellik | Detay |
    |---------|-------|
    | **Model** | Keras Sequential |
    | **Katmanlar** | Dense(128) → Dense(64) → Dense(1) |
    | **Vectorizer** | CountVectorizer (20.000 özellik) |
    | **Doğruluk** | ~%92.3 (doğrulama) |
    | **Görev** | İkili Sınıflandırma |
    """)

    st.markdown("---")
    st.markdown("""
    ```
       ___________________
      /                   \\
     /   T R U T H   or   \\
    |                      |
    |   F I C T I O N ?    |
    |                      |
     \\    S E N   K A R   /
      \\   V E R !        /
       \\________________/
              | |
              | |
           ___| |___
          |_________|
    ```
    """)


# ── Ana İçerik ──────────────────────────────────────────────────
st.markdown("### ✍️ Haber Metnini Girin")
news_text = st.text_area(
    label="Analiz edilecek haber metni:",
    height=200,
    placeholder="Haber metnini buraya yapıştırın...\n\nÖrnek: 'Scientists discover new method to detect misinformation using AI...'",
    label_visibility="collapsed",
)

col_btn1, col_btn2, col_btn3 = st.columns([2, 3, 2])
with col_btn2:
    analyze_btn = st.button(
        "🔍 HABERİ ANALİZ ET",
        use_container_width=True,
        type="primary",
    )

st.markdown('<hr class="newspaper-divider">', unsafe_allow_html=True)


# ── Sonuç Alanı ────────────────────────────────────────────────
if analyze_btn:
    if not news_text or news_text.strip() == "":
        st.warning("⚠️ Lütfen analiz etmek için bir haber metni girin.")
    else:
        with st.spinner("🔄 Haber analiz ediliyor... Yapay zekâ çalışıyor..."):
            model = load_model()
            vectorizer = load_vectorizer()

            if model is None:
                st.error(
                    "❌ Model dosyası bulunamadı: `fakeorreal_model.h5`\n\n"
                    "Lütfen `fakeorreal_model.h5` dosyasının uygulama dizininde "
                    "olduğundan emin olun."
                )
            elif vectorizer is None:
                st.error(
                    "❌ Vectorizer dosyası bulunamadı: `vectorizer.pkl`\n\n"
                    "Lütfen önce `save_vectorizer.py` ile vectorizer'ı kaydedin, "
                    "ardından `vectorizer.pkl` dosyasını uygulama dizinine koyun."
                )
            else:
                try:
                    is_real, real_score, fake_score = predict_news(news_text, model, vectorizer)

                    if is_real:
                        st.markdown(f"""
                        <div class="result-box-real">
                            <h2 style="margin:0; font-family: 'Playfair Display', serif;">
                                ✅ Sonuç: GERÇEK HABER
                            </h2>
                            <p class="confidence-label">
                                Güven Oranı: %{real_score * 100:.1f}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box-fake">
                            <h2 style="margin:0; font-family: 'Playfair Display', serif;">
                                🚨 Sonuç: SAHTE HABER
                            </h2>
                            <p class="confidence-label">
                                Güven Oranı: %{fake_score * 100:.1f}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("#### 📊 Güven Dağılımı")
                    col_r, col_f = st.columns(2)
                    with col_r:
                        st.metric(label="✅ Gerçek", value=f"%{real_score * 100:.1f}")
                        st.progress(real_score)
                    with col_f:
                        st.metric(label="❌ Sahte", value=f"%{fake_score * 100:.1f}")
                        st.progress(fake_score)

                except Exception as e:
                    st.error(f"❌ Tahmin sırasında bir hata oluştu: {e}")


# ── Örnek Haberler ──────────────────────────────────────────────
with st.expander("📑 Örnek Haberler ile Test Edin"):
    st.markdown("Aşağıdaki örneklerden birini kopyalayıp yukarıdaki alana yapıştırabilirsiniz:")

    st.markdown("**🟢 Gerçek Haber Örneği:**")
    st.code(
        "Washington (CNN) — The Federal Reserve raised its benchmark interest rate "
        "by a quarter of a percentage point on Wednesday, marking the tenth increase "
        "since March 2022 as the central bank continues its efforts to bring inflation "
        "back down to its 2% target. Fed Chair Jerome Powell said in a press conference "
        "that the labor market remains very tight and that inflation has moderated somewhat "
        "but remains well above the committee's longer-run goal.",
        language=None,
    )

    st.markdown("**🔴 Sahte Haber Örneği:**")
    st.code(
        "BREAKING: Scientists confirm drinking bleach cures all known diseases! "
        "The mainstream media is hiding this miracle cure from the public. "
        "Big Pharma doesn't want you to know this secret! Share before it gets deleted! "
        "The government has been suppressing this information for decades.",
        language=None,
    )


# ── Footer ──────────────────────────────────────────────────────
st.markdown("""
<div class="footer-newspaper">
    📰 The Truth Gazette &copy; 2026 &nbsp;|&nbsp; Developed with ❤️ by tugcesi
</div>
""", unsafe_allow_html=True)
