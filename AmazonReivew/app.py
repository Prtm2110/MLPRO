import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load the tokenizer and model
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model('sentiment_analysis_model.h5')

# Streamlit application
st.title('Sentiment Analysis of Musical Instrument Reviews')

review_text = st.text_area("Enter a review:")

if st.button('Predict Sentiment'):
    if review_text:
        # Preprocess the input review
        sequence = tokenizer.texts_to_sequences([review_text])
        padded_sequence = pad_sequences(sequence, maxlen=500)
        
        # Predict sentiment
        prediction = model.predict(padded_sequence)
        sentiment = 'Positive' if prediction[0] > 0.5 else 'Negative'
        
        st.write(f'Sentiment: **{sentiment}**')
    else:
        st.write("Please enter a review text.")
