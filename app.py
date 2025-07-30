import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer

@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

tokenizer, model = load_model()

def correct_grammar(text):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

st.title("📝 Grammar Corrector using BART")
user_input = st.text_area("Enter your sentence:")

if st.button("Correct Grammar"):
    if user_input.strip():
        corrected = correct_grammar(user_input)
        st.success("✅ Corrected:")
        st.write(corrected)
    else:
        st.warning("⚠️ Please enter a sentence.")





