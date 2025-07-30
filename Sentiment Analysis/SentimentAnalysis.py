import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import requests
import json

st.title("🐦 Mini Project: Analisis Sentimen Tweet Indonesia")
st.write("Lengkapi kode untuk membuat aplikasi analisis sentimen tweet!")

# Load model IndoBERT
@st.cache_resource
def load_indobert():
    model_name = "indolem/indobert-base-uncased"  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels= 3 
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    return tokenizer, model, device

LABEL_MAPPING = {
    0: 'negatif',  
    1: 'netral',  
    2: 'positif'   
}

# Fungsi untuk analisis sentimen
def analyze_sentiment(texts):
    tokenizer, model, device = load_indobert()
    results = []
    
    for text in texts:
        # Tokenize text
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True,
            padding=True,
            max_length=128  
        )
        
        # Pindahkan ke device yang tepat
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Prediksi sentimen
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_label].item()
        
        sentiment = LABEL_MAPPING[predicted_label] 
        
        results.append({
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'prob_negatif': probs[0][0].item(),
            'prob_netral': probs[0][1].item(),
            'prob_positif': probs[0][2].item()
        })
    
    return pd.DataFrame(results)

# Fungsi evaluasi dengan OLLAMA (evaluasi keseluruhan)
def evaluate_with_ollama(df_results, model="llama3.2"):
    total_tweets = len(df_results)
    sentiment_counts = df_results['sentiment'].value_counts()
    
    prompt_template = f"""
    Evaluasi hasil analisis sentimen Twitter Indonesia:
    
    Total tweets yang dianalisis: {total_tweets}
    
    Distribusi sentimen:
    [0  - negatif]
    [1  - netral]
    [2  - positif]
    
    Contoh hasil klasifikasi:
    [Baru coba sabun wajah ini, wanginya enak banget dan bikin kulit halus! - positif]
    [Packing oke, pengiriman cepat, barang sesuai deskripsi. Lumayan lah. - netral]
    [Produk ini gak sesuai ekspektasi, teksturnya aneh dan bikin lengket. - negatif]
    [Suka banget sama kualitas bahannya, adem dan jahitannya rapi! - positif]
    [Udah coba seminggu, belum kelihatan hasil yang signifikan sih. - netral]
    [Kecewa banget, baru dipakai 2 hari udah rusak. Gak recommended. - negatif]

    
    Pertanyaan evaluasi:
    1. Apakah distribusi sentimen masuk akal untuk data Twitter Indonesia?
    2. Apakah klasifikasi sudah akurat untuk bahasa gaul/informal?
    3. Berikan skor kualitas analisis (1-10) dan alasannya
    4. Apa saran perbaikan untuk model ini?
    """
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt_template,
                "stream": False
            }
        )
        return response.json()['response']
    except Exception as e:
        return f"Error: {str(e)}"

# Helper function untuk membaca CSV
def read_csv_flexible(uploaded_file):
    for delimiter in [',', ';', '\t']:
        for encoding in ['utf-8', 'latin1', 'ISO-8859-1']:
            try:
                return pd.read_csv(uploaded_file, delimiter=delimiter, encoding=encoding)
            except:
                continue
    return pd.DataFrame()



# Interface Streamlit
tab1, tab2, tab3 = st.tabs(["📝 Data Input", "🔍 Analisis", "📊 Evaluasi"])

with tab1:
    st.header("Input Data Tweet")
    
    sample_data = pd.DataFrame({
    'text': [
        "Baru coba sabun wajah ini, wanginya enak banget dan bikin kulit halus!",  # positif
        "Packing oke, pengiriman cepat, barang sesuai deskripsi. Lumayan lah.",    # netral
        "Produk ini gak sesuai ekspektasi, teksturnya aneh dan bikin lengket.",     # negatif
        "Suka banget sama kualitas bahannya, adem dan jahitannya rapi!",            # positif
        "Udah coba seminggu, belum kelihatan hasil yang signifikan sih.",           # netral
        "Kecewa banget, baru dipakai 2 hari udah rusak. Gak recommended.",          # negatif
        "Design produknya keren, cocok buat daily use. Value for money!",           # positif
        "Barang datang lengkap, tapi belum sempat dicoba. Semoga bagus.",           # netral
        "Serum ini bikin breakout parah, nyesel banget beli!",                      # negatif
        "Teknologi smart-nya beneran membantu. Simpel & efektif banget dipakai!"    # positif
    ]
})


    
    data_source = st.radio("Sumber data:", ["Sample Data", "Upload CSV"])
    
    if data_source == "Sample Data":
        st.write("Data tweet untuk analisis:")
        df = st.data_editor(sample_data, num_rows="dynamic")
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File berhasil dibaca!")
            except:
                st.error("Error membaca file. Implementasi read_csv_flexible() untuk handling yang lebih baik")
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()
    
    # Display data
    if not df.empty:
        st.dataframe(df)
        st.info(f"Total tweets: {len(df)}")

    st.session_state['data'] = df

with tab2:
    st.header("Analisis Sentimen Tweet")
    
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        # Implementasi tombol analisis
        if st.button("Analisis Sentimen", type="primary"):
            with st.spinner("Menganalisis tweet..."):
                results = analyze_sentiment(df['Text Tweet'].tolist())
                st.session_state['results'] = results
                
                st.dataframe(results)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Visualisasi distribusi sentimen
                    st.subheader("Distribusi Sentimen")
                    st.bar_chart(results['sentiment'].value_counts())
                    pass
                
                with col2:
                    # Visualisasi rata-rata probabilitas
                    st.subheader("Rata-rata Probabilitas")
                    avg_probs = results[['prob_negatif', 'prob_netral', 'prob_positif']].mean()
                    st.bar_chart(avg_probs)
                    pass

with tab3:
    st.header("Evaluasi dengan OLLAMA")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Model selection
        ollama_model = st.selectbox(
            "Model OLLAMA:",
            ["llama3.2", "llama3.1"],
            index=0
        )
        
        # Evaluasi keseluruhan
        if st.button("Evaluasi Dataset"):
            with st.spinner("Evaluasi dengan OLLAMA..."):
                evaluation = evaluate_with_ollama(results, ollama_model)
                
                st.subheader("Hasil Evaluasi:")
                st.text_area("", evaluation, height=300)
        
        if st.button("Download Hasil CSV"):
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download",
                data=csv, # Data CSV
                file_name="hasil_sentimen.csv",
                mime="text/csv"
            )

with st.sidebar:
    st.header("📚 Panduan Mini Project")
    st.write("""
    **Tugas Anda:**
    1. Load model IndoBERT
    2. Definisikan label mapping
    3. Implementasi analisis sentimen
    4. Buat evaluasi OLLAMA
    5. Siapkan data tweet Indonesia
    6. Implementasi analisis
    7. Tambahkan visualisasi
    8. Evaluasi dataset
    9. Fitur download
    10. Lengkapi dokumentasi
    
    **Model Info:**
    - Base: IndoBERT
    - Task: Tweet sentiment analysis
    - Classes: Positif, Netral, Negatif
    
    **Tips:**
    - Tweet max 128 tokens
    - Gunakan bahasa informal/gaul
    - Perhatikan confidence score
    """)
    
    st.header("🔧 Status")