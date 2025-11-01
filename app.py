
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re

# Configuration minimale
st.set_page_config(page_title="MentalGuard AI", layout="centered")

# Titre simple
st.title("🧠 MentalGuard AI")
st.write("Analyse de bien-être émotionnel")

# Fonction de nettoyage
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(text.split())

# Interface
user_text = st.text_area("Entrez votre texte:", height=100)

if st.button("Analyser"):
    if user_text.strip():
        try:
            # Charger le modèle
            model = tf.keras.models.load_model('mon_modele_depression_final.h5')
            with open('tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
            
            # Prétraitement
            text_clean = clean_text(user_text)
            sequence = tokenizer.texts_to_sequences([text_clean])
            sequence_padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100, padding='post')
            
            # Prédiction
            prediction = model.predict(sequence_padded, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            
            # Résultat
            st.success(f"Résultat: Niveau {predicted_class}")
            st.info(f"Confiance: {confidence:.1%}")
            
        except Exception as e:
            st.error(f"Erreur: {str(e)}")
    else:
        st.warning("Veuillez entrer du texte")
