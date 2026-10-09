# NLP Studio

A Streamlit application for text classification, summarization, English-to-French translation and text generation, supported by data preparation and model-training notebooks.

## Problem and solution

NLP Studio brings several text-processing tasks into one interface. PyTorch and Hugging Face Transformers handle inference, while the notebooks provide dataset preparation and training code. Three local model folders are required before the app can start; trained weights are not committed here.

## Architecture and technologies

| File | Role |
|---|---|
| `data_processor.ipynb` | Prepare IMDb classification, CNN/DailyMail summarization and WMT16 German-to-English translation data |
| `model_trainer.ipynb` | DistilBERT classification, T5 summarization/translation training and pretrained GPT-2 setup |
| `app.py` | Load models and present four inference tasks using Streamlit |
| `requirements.txt` | Python dependencies |

**Stack:** Python, PyTorch, Transformers, Hugging Face datasets, Streamlit, pandas, scikit-learn, NLTK and ROUGE evaluation utilities.

## Installation

```bash
git clone https://github.com/zik4O4/text-ai-app.git
cd text-ai-app
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install jupyter sentencepiece
```

The devcontainer specifies Python 3.11. On Windows, activate `.venv\Scripts\Activate.ps1`. Package versions are not locked; compatibility with model/tokenizer files must be validated in your environment. GPU acceleration is optional for inference; training is substantially more expensive on CPU.

## Model requirements

Provide these directories at the repository root:

```text
models/
  classification/
  summarization/
  generation/
```

Each folder must contain model weights, configuration and tokenizer files in a format accepted by Transformers. If any folder is missing, the app stops before exposing its tasks, including translation. English-to-French translation uses `Helsinki-NLP/opus-mt-en-fr` and requires a first-run download.

The project's existing [external model/data folder](https://drive.google.com/drive/folders/1kAg0OC9PlwAYGyQW9_Ua0DsMcUX7Nzeb?usp=sharing) is retained as a reference. Its availability and contents have not been independently verified. Check access, files and licensing before relying on it.

## Usage

```bash
streamlit run app.py
```

Choose a task, enter text and start processing. Summarization requires at least ten words. Generation samples a continuation; outputs are model predictions, not factual guarantees.

For the notebook workflow:

```bash
jupyter notebook data_processor.ipynb
jupyter notebook model_trainer.ipynb
```

Run from the repository root so generated `data/` and `models/` paths align with the app. Inspect dataset download and training settings before executing all cells. The preparation notebook writes task-specific training/validation CSVs used by the trainer. There are no standalone `data_processor.py` or `model_trainer.py` scripts in this snapshot.

## Existing interface screenshot

<img width="848" height="663" alt="Existing NLP Studio interface screenshot" src="https://github.com/user-attachments/assets/f27d2067-306f-4a38-ad6c-1be41401e530" />

## Limitations

- The training notebook's default translation data are German-to-English; the app uses a separate English-to-French pretrained model.
- GPT-2 generation is configured from pretrained weights, rather than demonstrated custom generation fine-tuning.
- Model artifacts and datasets require external downloads; a clean clone alone is insufficient for inference.
- No reproducible accuracy, latency or deployment benchmark is claimed here.

## Authors and license

Zakariya Ben Kassi and Youssef Monir Idrissi. Original contact information is retained below for attribution:

- BEN KASSI ZAKARIYA: zakariya.benkassi@gmail.com
- YOUSSEF MONIR IDRESSI: youssefmouniridrissi04@gmail.com

No repository-level license has been selected. Dataset and pretrained-model terms apply separately.
