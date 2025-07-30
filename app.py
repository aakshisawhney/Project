import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pretrained model
@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

tokenizer, model = load_model()

def correct_grammar(text):
    input_text = "correct grammar: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

# Streamlit UI
st.set_page_config(page_title="Grammar Corrector", layout="centered")
st.title("✍️ Grammar Correction using T5")
st.markdown("Fix grammar in your sentences using a pretrained ML model.")

user_input = st.text_area("Enter a sentence with grammar issues:")

if st.button("Correct Grammar"):
    if user_input.strip():
        corrected = correct_grammar(user_input)
        st.success("✅ Corrected Sentence:")
        st.write(corrected)
    else:
        st.warning("⚠️ Please enter some text first.")




