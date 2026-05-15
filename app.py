import streamlit as st
import joblib
import re

@st.cache_resource
def load_model():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# --- PREPROCESSING ---
slang_dict = {
    'gak': 'tidak', 'ga': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak',
    'gk': 'tidak', 'enggak': 'tidak', 'tdk': 'tidak',
    'gakada': 'tidak ada', 'gaada': 'tidak ada',
    'banget': 'sangat', 'bgt': 'sangat', 'bngt': 'sangat',
    'udah': 'sudah', 'udh': 'sudah', 'dah': 'sudah',
    'lagi': 'sedang', 'lg': 'sedang',
    'mau': 'ingin', 'mo': 'ingin', 'pengen': 'ingin', 'pgn': 'ingin',
    'buat': 'untuk', 'bwt': 'untuk', 'utk': 'untuk',
    'kalo': 'kalau', 'klo': 'kalau', 'klw': 'kalau',
    'trus': 'terus', 'trs': 'terus',
    'emang': 'memang', 'emg': 'memang',
    'kayak': 'seperti', 'kyk': 'seperti',
    'gimana': 'bagaimana', 'gmn': 'bagaimana',
    'kenapa': 'mengapa', 'knp': 'mengapa',
    'ntar': 'nanti', 'tar': 'nanti',
    'abis': 'habis',
    'nih': 'ini', 'tuh': 'itu',
    'doang': 'saja', 'aja': 'saja',
    'jg': 'juga', 'jga': 'juga',
    'dgn': 'dengan', 'dg': 'dengan',
    'yg': 'yang', 'yng': 'yang',
    'dr': 'dari', 'drpd': 'daripada',
    'krn': 'karena', 'karna': 'karena',
    'tp': 'tapi', 'ttpi': 'tapi',
    'sm': 'sama', 'ama': 'sama',
    'skrg': 'sekarang',
    'blm': 'belum', 'blom': 'belum',
    'hrs': 'harus', 'msti': 'mesti',
    'bs': 'bisa', 'bsa': 'bisa',
    'lbh': 'lebih', 'bbrp': 'beberapa',
    'seneng': 'senang',
    'capek': 'lelah', 'cape': 'lelah',
    'kesel': 'kesal', 'bosen': 'bosan',
    'males': 'malas',
    'mantap': 'bagus', 'mantep': 'bagus',
    'keren': 'bagus', 'jelek': 'buruk',
    'nyebelin': 'menjengkelkan', 'sebel': 'kesal',
    'kangen': 'rindu', 'sumpah': 'sungguh',
    'gue': 'saya', 'gw': 'saya', 'aku': 'saya',
    'lo': 'kamu', 'lu': 'kamu', 'elo': 'kamu',
    'doi': 'dia', 'dy': 'dia', 'mrk': 'mereka',
    'wkwk': '', 'wkwkwk': '', 'haha': '', 'hehe': '', 'hihi': '',
    'wah': '', 'waduh': '', 'duh': '', 'aduh': '',
    'eh': '', 'ah': '', 'oh': '', 'ih': '',
    'ok': 'baik', 'oke': 'baik',
    'btw': 'omong-omong', 'ni': 'ini',
    'yuk': 'mari', 'yukk': 'mari',
}

stopwords_id = {
    'dan', 'atau', 'tapi', 'namun', 'tetapi', 'jika', 'maka',
    'yang', 'di', 'ke', 'dari', 'pada', 'dengan', 'oleh', 'akan',
    'untuk', 'hingga', 'sampai', 'tentang', 'terhadap', 'dalam',
    'saat', 'ketika', 'karena', 'sebab', 'supaya', 'agar', 'bahwa',
    'sejak', 'selama', 'antara', 'melalui', 'setelah', 'sebelum',
    'sehingga', 'walaupun', 'meskipun', 'apabila', 'kalau',
    'ini', 'itu', 'sini', 'situ', 'sana',
    'kami', 'kita', 'dia', 'kamu', 'saya', 'anda', 'mereka',
    'adalah', 'ialah', 'para', 'si',
    'apa', 'siapa', 'mengapa', 'bagaimana', 'kapan', 'berapa',
    'lalu', 'kemudian', 'selain', 'sedangkan', 'padahal', 'sementara',
}

def clean_tweet(text):
    text = str(text).lower().strip()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'_', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_slang(text):
    tokens = text.split()
    normalized = [slang_dict.get(word, word) for word in tokens]
    return ' '.join([w for w in normalized if w != ''])

def remove_stopwords(text):
    tokens = text.split()
    return ' '.join([w for w in tokens if w not in stopwords_id and len(w) > 1])

def prediksi_sentimen(kalimat):
    processed = clean_tweet(kalimat)
    processed = normalize_slang(processed)
    processed = remove_stopwords(processed)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    proba = dict(zip(model.classes_, model.predict_proba(vec)[0]))
    return pred, proba, processed

# --- UI ---
st.set_page_config(page_title="Analisis Sentimen", page_icon="💬", layout="centered")
st.title("💬 Analisis Sentimen Tweet")
st.markdown("Masukkan teks tweet berbahasa Indonesia untuk diprediksi sentimennya.")

input_text = st.text_area("Masukkan teks:", height=120,
                           placeholder="Contoh: filmnya sangat mengecewakan dan membosankan")

if st.button("Prediksi Sentimen", type="primary"):
    if input_text.strip():
        pred, proba, processed = prediksi_sentimen(input_text)

        emoji_map = {'positif': '🟢', 'netral': '🟡', 'negatif': '🔴'}
        color_map = {'positif': 'green', 'netral': 'orange', 'negatif': 'red'}

        st.markdown(f"### Hasil: {emoji_map[pred]} **:{color_map[pred]}[{pred.upper()}]**")

        with st.expander("Lihat proses preprocessing"):
            st.write(f"**Teks asli:** {input_text}")
            st.write(f"**Setelah preprocessing:** {processed}")

        st.markdown("**Probabilitas per kelas:**")
        for label in ['positif', 'netral', 'negatif']:
            st.progress(proba[label], text=f"{label.capitalize()}: {proba[label]*100:.1f}%")
    else:
        st.warning("⚠️ Masukkan teks terlebih dahulu. ⚠️")
