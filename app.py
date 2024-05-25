import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

languages = {'French': '>>fra<< ', 'Italian': '>>ita<< ', 'Spanish': '>>spa<< ', 'Romanian': '>>ron<< ', 'Portuguese': '>>port<< '}

def clear_text():
    st.session_state["text"] = ""

@st.cache_resource()
def load_model_and_tokenizer():
    checkpoint = 'Klarly/multilingual-MT_Medical-Diagnostics_ROM'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    return tokenizer, model

st.set_page_config(
    page_title="Translator",
    page_icon=":globe_with_meridians:",
    layout="centered"  # Optional
)

st.title("Translator:globe_with_meridians:")

with st.container(border=True):

    col1, col2 = st.columns(2)  # Divide the page into two columns

    with col1:
        option1 = st.selectbox('**Source language**', ('English',))
        source_text = st.text_area("**Source text**", value="", height=None, max_chars=None, key="text")
        st.write('\n')
        translate_button = st.button("**Translate**", type="primary")

    with col2:
        option2 = st.selectbox('**Target language**', ('French', 'Italian', 'Portuguese', 'Spanish', 'Romanian'))
        target_text = st.empty()
        target_text_area = target_text.text_area("**Target text**", value='', height=None, max_chars=None, key=None)
        st.write('\n')
        st.button("**Clear**", on_click=clear_text)

tokenizer, model = load_model_and_tokenizer()

if translate_button:
    with st.spinner("Translating..."):
        target_language = option2.lower()
        prefix = languages[option2]
        input_text = prefix + source_text.strip()

        if input_text:
            inputs = tokenizer.encode(input_text, return_tensors="pt")
            outputs = model.generate(inputs, max_length=80, do_sample=True, top_k=30, top_p=0.95)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            target_text.text_area("**Target text**", value=translated_text, height=None, max_chars=None, key=None)
