# End-to-End NLP text-ia-app with Streamlit Deployment

This project demonstrates an end-to-end deep learning pipeline for text processing using NLP techniques. It includes data processing, model training, and deployment of the trained model via a Streamlit web app.

---

## Project Structure
```

project-root/
├── app.py                   # Streamlit app
├── data_processor.ipynb     # Notebook to process and prepare data
├── model_trainer.ipynb      # Notebook to train the model
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── data/                    # Folder containing raw dataset files (not included in repo)
└── models/                  # Folder containing pre-trained model files (not included in repo)

````
---

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/text-ai-app/.git
cd your-repo-name
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
````

3. **Prepare the data**

Run the data processing notebook or script to prepare the dataset:

```bash
# Option 1: Run the Jupyter notebook interactively
jupyter notebook data_processor.ipynb

# Option 2: Run the Python script (if available)
python data_processor.py
```

4. **Train the model (optional)**

If you want to train the model yourself, run the training notebook or script:

```bash
jupyter notebook model_trainer.ipynb
# or
python model_trainer.py
```

> **Note:** Training can be time-consuming. raw dataset files and Pre-trained models are available for download (see below).

---

## Download Pre-trained Models

The raw dataset files and pre-trained model files (~1GB) are available here:

[Download models from Google Drive](https://drive.google.com/drive/folders/1kAg0OC9PlwAYGyQW9_Ua0DsMcUX7Nzeb?usp=sharing)

### Instructions

1. Download and extract the models folder to the root directory of the project.
2. Ensure the folder structure looks like:

```
project-root/
├── app.py
├── models/
│   ├── classification
│       ├── config.json
│       ├── pytorch_model.bin
│       └── ...
│  ├──generation
│     ├── config.json
│     ├── pytorch_model.bin
│     └── ...
│  └── ... 
│   
```

---

## Running the Streamlit App

Once dependencies are installed and models are in place, start the app with:

```bash
streamlit run app.py
```

Open your browser to the local URL provided to interact with the app.

---

## Additional Notes

* The project uses Hugging Face Transformers and PyTorch for NLP tasks such as text summarization, classification, or generation.
* Required NLTK datasets can be installed by running the following in a Python shell or notebook:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## Contact

BEN KASSI ZAKARIYA— [zakariya.benkassi@gmail.com]
YOUSSEF MONIR IDRESSI [ youssefmouniridrissi04@gmail.com ]
Project Link: [https://github.com/your-username/your-repo-name]


