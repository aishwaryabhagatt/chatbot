import streamlit as st
import json
import numpy as np
import pickle
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

# Load the intents file
with open('intents.json') as file:
    data = json.load(file)

@st.cache_resource
def load_model():
    return keras.models.load_model('chat-model')

model = load_model()

# Load tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load label encoder object
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

max_len = 20

st.title("Pandora: Your Personal Therapeutic AI Assistant")
st.write("Start talking with Pandora. Type 'quit' to stop.")

user_input = st.text_input("You:", key="user_input")

if user_input:
    if user_input.lower() == 'quit':
        st.write("Pandora: Take care. See you soon.")
    else:
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]), truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                response = np.random.choice(i['responses'])
                st.write(f"Pandora: {response}")
                break
