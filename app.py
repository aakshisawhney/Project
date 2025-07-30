import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
    model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")
    return tokenizer, model

tokenizer, model = load_model()

def correct_grammar(text):
    input_text = f"gec: {text}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs, max_length=128, num_beams=5, early_stopping=True)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

st.set_page_config(page_title="Grammar Corrector", layout="centered")
st.title("📝 Grammar Error Corrector")
st.markdown("Enter a sentence with grammar mistakes:")

text_input = st.text_area("Your sentence:")

if st.button("Correct Grammar"):
    if text_input.strip():
        result = correct_grammar(text_input)
        st.success("✅ Corrected Sentence:")
        st.write(result)
    else:
        st.warning("⚠️ Please enter some text.")






