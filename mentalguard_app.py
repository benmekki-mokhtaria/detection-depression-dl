
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re

# Configuration de base
st.set_page_config(
    page_title="MentalGuard AI",
    page_icon="🧠",
    layout="centered"
)

# Style simple
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2em;
    }
</style>
""", unsafe_allow_html=True)

# Titre
st.markdown('<div class="main-title">🧠 MentalGuard AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyse de bien-être émotionnel</div>', unsafe_allow_html=True)

# Fonction de nettoyage
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Correction des séquences d'échappement
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Chargement du modèle avec gestion d'erreur
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('mon_modele_depression_final.h5')
        return model
    except Exception as e:
        st.error(f"Erreur chargement modèle: {e}")
        return None

@st.cache_resource
def load_tokenizer():
    try:
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer
    except Exception as e:
        st.error(f"Erreur chargement tokenizer: {e}")
        return None

# Interface
text_input = st.text_area(
    "Entrez votre texte à analyser:",
    height=120,
    placeholder="Exemple: Je me sens bien aujourd'hui..."
)

if st.button("Analyser le texte", type="primary"):
    if text_input.strip():
        with st.spinner("Analyse en cours..."):
            # Charger modèle et tokenizer
            model = load_model()
            tokenizer = load_tokenizer()
            
            if model and tokenizer:
                try:
                    # Nettoyage
                    text_clean = clean_text(text_input)
                    
                    # Tokenization
                    sequence = tokenizer.texts_to_sequences([text_clean])
                    sequence_padded = tf.keras.preprocessing.sequence.pad_sequences(
                        sequence, maxlen=100, padding='post'
                    )
                    
                    # Prédiction
                    prediction = model.predict(sequence_padded, verbose=0)
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    confidence = np.max(prediction)
                    
                    # Affichage résultat
                    st.success("✅ Analyse terminée!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Niveau détecté", f"Classe {predicted_class}")
                    with col2:
                        st.metric("Confiance", f"{confidence:.1%}")
                        
                    # Interprétation
                    interpretations = [
                        "🟢 Bien-être optimal",
                        "🟡 Léger malaise", 
                        "🟠 Signes modérés",
                        "🔴 Signes importants",
                        "⚫ Consultation recommandée"
                    ]
                    
                    st.info(f"**Interprétation:** {interpretations[predicted_class]}")
                    
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse: {e}")
            else:
                st.error("❌ Impossible de charger le modèle ou le tokenizer")
    else:
        st.warning("⚠️ Veuillez entrer un texte à analyser")

# Pied de page
st.markdown("---")
st.caption("MentalGuard AI • Outil d'analyse émotionnelle")
