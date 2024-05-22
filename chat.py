import json
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the intents file
with open('intents.json') as file:
    data = json.load(file)

@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('chat-model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

@st.cache_resource
def load_label_encoder():
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
    return lbl_encoder

# Load the model and other files
model = load_model()
tokenizer = load_tokenizer()
lbl_encoder = load_label_encoder()

# Check if the model is loaded successfully
if model is None:
    st.stop()

# Parameters
max_len = 20

st.title("Pandora - Your Personal Therapeutic AI Assistant")

def chat():
    st.write("Start talking with Pandora (type 'quit' to stop).")
    user_input = st.text_input("You: ", "")
    if user_input.lower() == 'quit':
        st.write("Pandora: Take care. See you soon.")
    elif user_input:
        with st.spinner("Pandora is thinking..."):
            result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]), truncating='post', maxlen=max_len))
            tag = lbl_encoder.inverse_transform([np.argmax(result)])
            for i in data['intents']:
                if i['tag'] == tag:
                    st.write("Pandora: " + np.random.choice(i['responses']))

chat()
