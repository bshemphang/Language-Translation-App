import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

st.title('Language Translation App')

src_text = st.text_area("Enter text to translate:")

lang_options = {
    'French': 'fr',
    'German': 'de',
    'Spanish': 'es'
}
tgt_lang = st.selectbox("Select target language:", list(lang_options.keys()))

if st.button("Translate"):
    if src_text and tgt_lang:
        model_name = f'Helsinki-NLP/opus-mt-en-{lang_options[tgt_lang]}'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        inputs = tokenizer(src_text, return_tensors="pt", padding=True)

        translated = model.generate(**inputs)
        tgt_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        st.write(f"Translated text: {tgt_text}")
    else:
        st.write("Please enter text and select a target language.")
