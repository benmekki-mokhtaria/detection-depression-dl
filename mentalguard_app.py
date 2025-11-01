
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re

st.set_page_config(page_title="MentalGuard AI", layout="wide")

st.markdown("<h1 style='text-align: center;'>🛡️ MentalGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Système d'Analyse Émotionnelle</p>", unsafe_allow_html=True)

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('mon_modele_depression_final.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except:
        return None, None

user_text = st.text_area("Entrez votre texte:", height=150)

if st.button("Analyser"):
    if user_text.strip():
        model, tokenizer = load_model()
        if model and tokenizer:
            text_clean = clean_text(user_text)
            sequence = tokenizer.texts_to_sequences([text_clean])
            sequence_padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100, padding='post')
            prediction = model.predict(sequence_padded, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            st.success(f"Résultat: Classe {predicted_class} (Confiance: {confidence:.1%})")
