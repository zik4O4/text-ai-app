import streamlit as st
from transformers import pipeline
import torch
import os

# Function to check if environment variables for custom models exist
def get_model_id(env_var_name, default_model):
    model_id = os.environ.get(env_var_name, default_model)
    return model_id

@st.cache_resource
def load_models():
    # Use well-established models instead of custom ones that are failing
    # You can set environment variables to override these defaults
    classifier_id = get_model_id("CLASSIFIER_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
    summarizer_id = get_model_id("SUMMARIZER_MODEL", "facebook/bart-large-cnn")
    generation_id = get_model_id("GENERATOR_MODEL", "gpt2")
    translation_id = get_model_id("TRANSLATOR_MODEL", "Helsinki-NLP/opus-mt-en-fr")
    
    # Determine device (GPU or CPU)
    device = 0 if torch.cuda.is_available() else -1
    st.sidebar.info(f"Using {'GPU' if device == 0 else 'CPU'} for inference")
    
    # Show model loading status
    models_status = st.sidebar.empty()
    models_status.info("Loading models... This may take a few minutes on first run.")
    
    # Load classification model
    try:
        models_status.info("Loading classification model...")
        classifier = pipeline("text-classification", model=classifier_id, device=device)
        st.sidebar.success(f"✅ Classification model loaded: {classifier_id}")
    except Exception as e:
        st.sidebar.error(f"Failed to load classification model: {str(e)[:100]}...")
        classifier = None
    
    # Load summarization model
    try:
        models_status.info("Loading summarization model...")
        summarizer = pipeline("summarization", model=summarizer_id, device=device)
        st.sidebar.success(f"✅ Summarization model loaded: {summarizer_id}")
    except Exception as e:
        st.sidebar.error(f"Failed to load summarization model: {str(e)[:100]}...")
        summarizer = None
    
    # Load text generation model
    try:
        models_status.info("Loading text generation model...")
        generator = pipeline("text-generation", model=generation_id, device=device)
        st.sidebar.success(f"✅ Text generation model loaded: {generation_id}")
    except Exception as e:
        st.sidebar.error(f"Failed to load generation model: {str(e)[:100]}...")
        generator = None
    
    # Load translation model
    try:
        models_status.info("Loading translation model...")
        # Make sure sentencepiece is installed for translation models
        try:
            import sentencepiece
        except ImportError:
            st.sidebar.warning("sentencepiece is not installed. Installing it may fix translation model issues.")
        translator = pipeline("translation_en_to_fr", model=translation_id, device=device)
        st.sidebar.success(f"✅ Translation model loaded: {translation_id}")
    except Exception as e:
        st.sidebar.error(f"Failed to load translation model: {str(e)[:100]}...")
        translator = None
    
    # Clear loading message
    models_status.empty()
    
    return classifier, summarizer, translator, generator

# Display app title and info
st.title("ZyNLP - Hugging Face Powered NLP App")
st.markdown("""
This app uses pre-trained models from Hugging Face to perform various NLP tasks.
Select a task from the dropdown and enter your text to get started!
""")

# Sidebar for model information
st.sidebar.title("Model Information")
st.sidebar.markdown("Loading models from Hugging Face...")

# Load models
classifier, summarizer, translator, generator = load_models()

# Determine available models
models = {
    "Classification": classifier, 
    "Summarization": summarizer, 
    "Translation (EN → FR)": translator, 
    "Text Generation": generator
}
available_tasks = [task for task, model in models.items() if model is not None]

# Check if at least one model is available
if not available_tasks:
    st.error("No models could be loaded. Please check the sidebar for specific errors.")
    st.info("""
    ### Troubleshooting Tips:
    1. Make sure you have the required libraries installed: `pip install transformers torch sentencepiece`
    2. Check your internet connection
    3. Use well-established model names instead of custom ones
    """)
    st.stop()

# Main application UI
st.subheader("Step 1: Choose an NLP Task")
task = st.selectbox("Select task:", available_tasks)

st.subheader("Step 2: Enter Your Text")
input_text = st.text_area("Enter text here:", height=200)

# Add task-specific instructions
if task == "Classification":
    st.info("Classification will categorize your text. Works well with sentences or short paragraphs.")
elif task == "Summarization":
    st.info("Summarization works best with longer texts (paragraphs or articles).")
elif task == "Translation (EN → FR)":
    st.info("Enter English text to translate to French.")
elif task == "Text Generation":
    st.info("Enter a prompt and the model will generate text continuing from your input.")

# Advanced options for each task
with st.expander("Advanced Options"):
    if task == "Classification":
        # No additional options for classification
        pass
    elif task == "Summarization":
        max_length = st.slider("Maximum summary length", 30, 500, 100)
        min_length = st.slider("Minimum summary length", 10, 100, 30)
    elif task == "Translation (EN → FR)":
        # No additional options for translation
        pass
    elif task == "Text Generation":
        max_length = st.slider("Maximum length", 10, 500, 100)
        temperature = st.slider("Temperature (higher = more creative)", 0.1, 1.5, 0.7)
        top_p = st.slider("Top-p (nucleus sampling)", 0.5, 1.0, 0.9)

# Process the input when the user clicks the Run button
if st.button("Run"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner(f"Running {task}..."):
            try:
                if task == "Classification":
                    outputs = classifier(input_text)
                    # Create a DataFrame for better visualization
                    import pandas as pd
                    if isinstance(outputs, list):
                        df = pd.DataFrame(outputs)
                        # Convert scores to percentages
                        if 'score' in df.columns:
                            df['confidence'] = df['score'].apply(lambda x: f"{x*100:.2f}%")
                        st.dataframe(df)
                    else:
                        st.write(outputs)
                    
                    # Show the top prediction prominently
                    label = outputs[0]['label']
                    score = outputs[0]['score']
                    st.success(f"Prediction: **{label}** (confidence: {score:.2f})")
                
                elif task == "Summarization":
                    if len(input_text.split()) < 10:
                        st.warning("Summarization works best for longer text.")
                    
                    # Get advanced options if available
                    ml = max_length if 'max_length' in locals() else 100
                    minl = min_length if 'min_length' in locals() else 30
                    
                    with st.spinner("Generating summary..."):
                        summary = summarizer(input_text, max_length=ml, min_length=minl, do_sample=False)[0]['summary_text']
                        st.success("Summary:")
                        st.write(summary)
                
                elif task == "Translation (EN → FR)":
                    with st.spinner("Translating..."):
                        translation = translator(input_text)[0]['translation_text']
                        st.success("Translation (EN → FR):")
                        st.write(translation)
                
                elif task == "Text Generation":
                    # Get advanced options if available
                    ml = max_length if 'max_length' in locals() else 100
                    temp = temperature if 'temperature' in locals() else 0.7
                    tp = top_p if 'top_p' in locals() else 0.9
                    
                    with st.spinner("Generating text..."):
                        result = generator(
                            input_text, 
                            max_length=len(input_text.split()) + ml, 
                            num_return_sequences=1, 
                            do_sample=True, 
                            temperature=temp,
                            top_p=tp
                        )[0]['generated_text']
                        
                        # Only show the newly generated part
                        if result.startswith(input_text):
                            new_text = result[len(input_text):]
                            st.success("Generated Continuation:")
                            st.write(new_text)
                            
                            # Show the full text in an expander
                            with st.expander("Show full text"):
                                st.write(result)
                        else:
                            st.success("Generated Text:")
                            st.write(result)
            
            except Exception as e:
                st.error(f"Error during {task}: {str(e)}")
                st.info("Try with a different input text or select another task.")
