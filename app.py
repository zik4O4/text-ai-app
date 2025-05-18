import os
import streamlit as st
from transformers import pipeline, MarianMTModel, MarianTokenizer
import torch

@st.cache_resource
def load_models():
    # Define model directories
    classifier_dir = "./models/classification"
    summarizer_dir = "./models/summarization"
    generation_dir = "./models/generation"
    
    # Choose a specific MarianMT model from Hugging Face
    translation_model_name = "Helsinki-NLP/opus-mt-en-fr"  # English to French translation
    
    # Check if local directories exist
    for d, name in zip([classifier_dir, summarizer_dir, generation_dir], 
                     ["classification", "summarization", "generation"]):
        if not os.path.exists(d):
            st.error(f"Model directory not found: {d}")
            return None, None, None, None

    # Determine device
    device = 0 if torch.cuda.is_available() else -1

    # Load classification pipeline
    classifier = pipeline(
        "text-classification",
        model=classifier_dir,
        tokenizer=classifier_dir,
        device=device
    )

    # Load summarization pipeline
    summarizer = pipeline(
        "summarization",
        model=summarizer_dir,
        tokenizer=summarizer_dir,
        device=device
    )

    # Load translation pipeline using a pre-trained model from Hugging Face
    try:
        translator = pipeline(
            "translation",
            model=translation_model_name,
            device=device
        )
    except Exception as e:
        st.error(f"Failed to load translation model: {e}")
        translator = None

    # Load text generation pipeline
    generator = pipeline(
        "text-generation",
        model=generation_dir,
        tokenizer=generation_dir,
        device=device
    )

    return classifier, summarizer, translator, generator


classifier, summarizer, translator, generator = load_models()

# Check if any model failed to load
failed_models = []
if classifier is None: failed_models.append("Classification")
if summarizer is None: failed_models.append("Summarization")
if translator is None: failed_models.append("Translation")
if generator is None: failed_models.append("Text Generation")

if failed_models:
    st.warning(f"Some models failed to load: {', '.join(failed_models)}")
    st.info("You can still use the available models.")
    available_tasks = ["Classification", "Summarization", "Translation", "Text Generation"]
    available_tasks = [task for task, model in zip(available_tasks, [classifier, summarizer, translator, generator]) if model is not None]
else:
    available_tasks = ["Classification", "Summarization", "Translation", "Text Generation"]

if len(available_tasks) == 0:
    st.error("No models were loaded successfully. Please check your model directories.")
    st.stop()

st.title("ZY ")

task = st.selectbox("Choose an NLP task", available_tasks)

input_text = st.text_area("Enter text here:", height=200)

if st.button("Run"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner(f"Running {task}..."):
            try:
                if task == "Classification" and classifier is not None:
                    outputs = classifier(input_text)
                    label = outputs[0]['label']
                    score = outputs[0]['score']
                    st.success(f"Prediction: **{label}** (confidence: {score:.2f})")

                elif task == "Summarization" and summarizer is not None:
                    if len(input_text.split()) < 10:
                        st.warning("Summarization works best for texts longer than 10 words.")
                    else:
                        summary = summarizer(input_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                        st.success("Summary:")
                        st.write(summary)

                elif task == "Translation" and translator is not None:
                    st.write("EN to FR")
                    translation = translator(input_text)[0]['translation_text']
                    st.success("Translation:")
                    st.write(translation)
                    

                elif task == "Text Generation" and generator is not None:
                    result = generator(input_text, max_length=100, num_return_sequences=1, do_sample=True, temperature=0.7)[0]['generated_text']
                    # Strip prompt prefix if present
                    if result.startswith(input_text):
                        result = result[len(input_text):].strip()
                    st.success("Generated Text:")
                    st.write(result)

            except Exception as e:
                st.error(f"Error during {task}: {e}")
                import traceback
                st.code(traceback.format_exc(), language="python")
