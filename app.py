@st.cache_resource
def load_models():
    # Hugging Face model IDs
    classifier_id = get_model_id("CLASSIFIER_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
    summarizer_id = get_model_id("SUMMARIZER_MODEL", "facebook/bart-large-cnn")
    generation_id = get_model_id("GENERATOR_MODEL", "gpt2")
    translation_id = "Helsinki-NLP/opus-mt-en-fr"

    device = 0 if torch.cuda.is_available() else -1

    try:
        classifier = pipeline("text-classification", model=classifier_id, tokenizer=classifier_id, device=device)
    except Exception as e:
        st.error(f"Failed to load classification model: {e}")
        classifier = None

    try:
        summarizer = pipeline("summarization", model=summarizer_id, tokenizer=summarizer_id, device=device)
    except Exception as e:
        st.error(f"Failed to load summarization model: {e}")
        summarizer = None

    try:
        generator = pipeline("text-generation", model=generation_id, tokenizer=generation_id, device=device)
    except Exception as e:
        st.error(f"Failed to load generation model: {e}")
        generator = None

    try:
        translator = pipeline("translation_en_to_fr", model=translation_id, device=device)
    except Exception as e:
        st.error(f"Failed to load translation model: {e}")
        translator = None

    return classifier, summarizer, translator, generator

classifier, summarizer, translator, generator = load_models()

# Determine available models
models = {"Classification": classifier, "Summarization": summarizer, "Translation": translator, "Text Generation": generator}
available_tasks = [task for task, model in models.items() if model is not None]

if not available_tasks:
    st.error("No models could be loaded. Please check Hugging Face model names.")
    st.stop()

# UI
st.title("ZyNLP - Hugging Face Powered NLP App")
task = st.selectbox("Choose an NLP task", available_tasks)
input_text = st.text_area("Enter text here:", height=200)

if st.button("Run"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner(f"Running {task}..."):
            try:
                if task == "Classification":
                    outputs = classifier(input_text)
                    label = outputs[0]['label']
                    score = outputs[0]['score']
                    st.success(f"Prediction: **{label}** (confidence: {score:.2f})")

                elif task == "Summarization":
                    if len(input_text.split()) < 10:
                        st.warning("Summarization works best for longer text.")
                    else:
                        summary = summarizer(input_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                        st.success("Summary:")
                        st.write(summary)

                elif task == "Translation":
                    translation = translator(input_text)[0]['translation_text']
                    st.success("Translation (EN → FR):")
                    st.write(translation)

                elif task == "Text Generation":
                    result = generator(input_text, max_length=100, num_return_sequences=1, do_sample=True, temperature=0.7)[0]['generated_text']
                    if result.startswith(input_text):
                        result = result[len(input_text):].strip()
                    st.success("Generated Text:")
                    st.write(result)

            except Exception as e:
                st.error(f"Error during {task}: {e}")

