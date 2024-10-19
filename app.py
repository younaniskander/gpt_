import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load Hugging Face model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400m-distill")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400m-distill")

# Streamlit UI setup
st.set_page_config(page_title="Conversational Q&A Chatbot with Hugging Face")
st.header("Hey, let's chat!")

# Function to get a response from the Hugging Face model
def get_chatmodel_response(question):
    # Tokenize the input
    inputs = tokenizer.encode(question, return_tensors="pt")

    # Generate a response using the model
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)

    # Decode the output to get the text response
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

# Input field for user to ask questions
input = st.text_input("Ask something:", key="input")
if input:
    response = get_chatmodel_response(f"the answer to the universe is {input}.")
    st.subheader("The response is:")
    st.write(response)