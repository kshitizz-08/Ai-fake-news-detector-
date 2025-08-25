import streamlit as st
import joblib

# Load your model
model = joblib.load("model.pkl")  # Replace with your model file
vectorizer = joblib.load("vectorizer.pkl")  # If you have one

st.title("📰 Fake News Detector")
st.write("Enter news text below and check if it's Fake or Real")

user_input = st.text_area("Enter News Text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Transform and predict
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        if prediction == 0:
            st.error("This news is **FAKE** ❌")
        else:
            st.success("This news is **REAL** ✅")
