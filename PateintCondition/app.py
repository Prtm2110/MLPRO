import streamlit as st
import pickle

# Load the trained model and vectorizer
with open('passive_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit app
st.title("Drug Review Sentiment Analysis")

# User input
user_input = st.text_area("Enter a drug review:", "")

if st.button("Predict Condition"):
    if user_input:
        # Transform the user input using the vectorizer
        transformed_input = vectorizer.transform([user_input])
        
        # Make a prediction
        prediction = model.predict(transformed_input)[0]
        
        # Display the prediction
        st.write(f"The predicted condition is: **{prediction}**")
    else:
        st.write("Please enter a review to make a prediction.")

st.write("This app predicts the condition (e.g., 'Depression', 'Acne', etc.) based on the text of a drug review.")
