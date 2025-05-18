import os
import streamlit as st
from transformers import pipeline
import torch
import gdown

@st.cache_resource
def load_models():
    # Define model directories
    base_dir = "./models"
    classifier_dir = os.path.join(base_dir, "classification")
    summarizer_dir = os.path.join(base_dir, "summarization")
    generation_dir = os.path.join(base_dir, "generation")

    # Google Drive file IDs
    gdrive_links = {
        "classification": "1vcsMiMTh3BiRzvIgtu4BlUyUuvLjSqdL",
        "summarization": "1kL5XsOHc4rxC8psh4kVoZkSnzTUM7NPH",
        "generation": "1pbo8sOWlSukxKqCKrDC7b4k1R4nnhEro"
    }

    os.makedirs(base_dir, exist_ok=True)

    # Download models if missing
    for model_type, folder in zip(["classification", "summarization", "generation"],
                                  [classifier_dir, summarizer_dir, generation_dir]):
        if not os.path.exists(folder):
            st.info(f"Downloading {model_type} model...")
            gdown.download_folder(f"https://drive.google.com/drive/folders/{gdrive_links[model_type]}", output=folder, quiet=False, use_cookies=False)

    # Load MarianMT model from Hugging Face (EN to FR)
    translation_model_name = "Helsinki-NLP/opus-mt-en-fr"

    device = 0 if torch.cuda.is_available() else -1

    try:
        classifier = pipeline("text-classification", model=classifier_dir, tokenizer=classifier_dir, device=device)
    except:
        classifier = None

    try:
        summarizer = pipeline("summarization", model=summarizer_dir, tokenizer=summarizer_dir, device=device)
    except:
        summarizer = None

    try:
        generator = pipeline("text-generation", model=generation_dir, tokenizer=generation_dir, device=device)
    except:
        generator = None

    try:
        translator = pipeline("translation_en_to_fr", model=translation_model_name, tokenizer=translation_model_name, device=device)
    except:
        translator = None

    return classifier, summarizer, translator, generator

classifier, summarizer, translator, generator = load_models()

# Check for loaded models
tasks = {
    "Classification": classifier,
    "Summarization": summarizer,
    "Translation (EN → FR)": translator,
    "Text Generation": generator
}

available_tasks = [name for name, model in tasks.items() if model is not None]

if not available_tasks:
    st.error("No models could be loaded.")
    st.stop()

st.title("NLP Pipeline App")

task = st.selectbox("Choose an NLP task", available_tasks)
text = st.text_area("Enter your text here", height=200)

if st.button("Run") and text.strip():
    with st.spinner(f"Running {task}..."):
        if task == "Classification":
            result = tasks[task](text)[0]
            st.success(f"Label: {result['label']} (Confidence: {result['score']:.2f})")

        elif task == "Summarization":
            summary = tasks[task](text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            st.success("Summary:")
            st.write(summary)

        elif task == "Translation (EN → FR)":
            translation = tasks[task](text)[0]['translation_text']
            st.success("Translation:")
            st.write(translation)

        elif task == "Text Generation":
            output = tasks[task](text, max_length=100, num_return_sequences=1)[0]['generated_text']
            st.success("Generated Text:")
            st.write(output)
else:
    if not text.strip():
        st.info("Please enter some text.")