import os
import streamlit as st
from transformers import pipeline
import torch

st.set_page_config(page_title="NLP Studio", page_icon="🧠", layout="centered")

st.markdown("""
<style>
/* FOND ANIMÉ EN GRADIENT */
body {
    background: linear-gradient(120deg, #d5e1f2, #f2f4f9, #e2f0f9);
    background-size: 400% 400%;
    animation: gradientShift 20s ease infinite;
}
@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* CONTENEUR GLASS EFFECT */
.main-container {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 20px;
    padding: 3rem;
    max-width: 850px;
    margin: auto;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    animation: fadeIn 1s ease-in-out;
}

@keyframes fadeIn {
    0% {opacity: 0; transform: translateY(20px);}
    100% {opacity: 1; transform: translateY(0);}
}

.title {
    font-size: 2.8rem;
    font-weight: bold;
    color: #1b263b;
    text-align: center;
    margin-bottom: 0.3rem;
}
.subtitle {
    font-size: 1.1rem;
    text-align: center;
    color: #4b5563;
    margin-bottom: 2rem;
}
.stTextArea label, .stSelectbox label {
    font-weight: bold;
    color: #2f3e46;
}
.stButton > button {
    background: linear-gradient(90deg, #3b82f6, #6366f1);
    color: white;
    font-size: 1rem;
    padding: 0.6rem 1.5rem;
    border: none;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    transition: background 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #2563eb, #4f46e5);
}
.result-box {
    background-color: #f9fafb;
    border-left: 5px solid #3b82f6;
    padding: 1rem;
    margin-top: 1.5rem;
    border-radius: 12px;
    color: #1f2937;
    font-size: 1rem;
    animation: fadeIn 0.5s ease-in-out;
}
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    classifier_dir = "./models/classification"
    summarizer_dir = "./models/summarization"
    generation_dir = "./models/generation"
    translation_model_name = "Helsinki-NLP/opus-mt-en-fr"

    for d in [classifier_dir, summarizer_dir, generation_dir]:
        if not os.path.exists(d):
            st.error(f"Model directory not found: {d}")
            return None, None, None, None

    device = 0 if torch.cuda.is_available() else -1

    classifier = pipeline("text-classification", model=classifier_dir, tokenizer=classifier_dir, device=device)
    summarizer = pipeline("summarization", model=summarizer_dir, tokenizer=summarizer_dir, device=device)
    try:
        translator = pipeline("translation", model=translation_model_name, device=device)
    except Exception as e:
        st.error(f"Translation model error: {e}")
        translator = None
    generator = pipeline("text-generation", model=generation_dir, tokenizer=generation_dir, device=device)

    return classifier, summarizer, translator, generator

classifier, summarizer, translator, generator = load_models()

failed_models = []
if classifier is None: failed_models.append("Classification")
if summarizer is None: failed_models.append("Summarization")
if translator is None: failed_models.append("Translation")
if generator is None: failed_models.append("Text Generation")

available_tasks = [t for t, m in zip(["Classification", "Summarization", "Translation", "Text Generation"], 
                                     [classifier, summarizer, translator, generator]) if m is not None]

if not available_tasks:
    st.error("❌ Aucun modèle n'a été chargé.")
    st.stop()

# --- UI PRINCIPALE ---
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="title">🧠 NLP Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Interface intelligente pour vos tâches NLP : classification, résumé, traduction, génération</div>', unsafe_allow_html=True)

    task = st.selectbox("📌 Sélectionnez une tâche :", available_tasks)
    input_text = st.text_area("✍️ Entrez votre texte ici :", height=200)

    if st.button("🚀 Lancer le traitement"):
        if not input_text.strip():
            st.warning("⛔ Le champ est vide.")
        else:
            with st.spinner(f"Traitement en cours pour {task}..."):
                try:
                    if task == "Classification":
                        result = classifier(input_text)[0]
                        st.markdown(f'<div class="result-box">🔍 <b>Label :</b> {result["label"]} <br>📊 <b>Confiance :</b> {result["score"]:.2f}</div>', unsafe_allow_html=True)

                    elif task == "Summarization":
                        if len(input_text.split()) < 10:
                            st.warning("⚠️ Veuillez entrer au moins 10 mots.")
                        else:
                            summary = summarizer(input_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                            st.markdown(f'<div class="result-box">📝 <b>Résumé :</b><br>{summary}</div>', unsafe_allow_html=True)

                    elif task == "Translation":
                        translation = translator(input_text)[0]['translation_text']
                        st.markdown(f'<div class="result-box">🌍 <b>Traduction (EN → FR) :</b><br>{translation}</div>', unsafe_allow_html=True)

                    elif task == "Text Generation":
                        generated = generator(input_text, max_length=100, num_return_sequences=1, do_sample=True, temperature=0.7)[0]['generated_text']
                        if generated.startswith(input_text):
                            generated = generated[len(input_text):].strip()
                        st.markdown(f'<div class="result-box">💬 <b>Texte généré :</b><br>{generated}</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"💥 Erreur : {e}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")

    st.markdown('</div>', unsafe_allow_html=True)  # close container
