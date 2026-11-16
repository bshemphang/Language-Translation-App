# Language Translation App

Translate English text into French, German, or Spanish with a small Streamlit app. It uses Hugging Face MarianMT models (Helsinki-NLP opus-mt) and runs the translation locally on your machine.

## Tech stack

- Python
- Streamlit
- Hugging Face Transformers (MarianMT)
- PyTorch

## Getting started

```bash
pip install streamlit transformers torch
streamlit run language_translation.py
```

Pick a target language, type your English text, and click Translate. The first run downloads the model for the language you pick.

## Supported languages

- French (Helsinki-NLP/opus-mt-en-fr)
- German (Helsinki-NLP/opus-mt-en-de)
- Spanish (Helsinki-NLP/opus-mt-en-es)
