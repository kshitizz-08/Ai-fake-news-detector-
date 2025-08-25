#!/usr/bin/env python3
"""
Test script to diagnose the fake news detection model
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def test_model():
    """Test the model with sample texts"""
    
    # Load the model and vectorizer
    try:
        with open('backend/model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
        
        with open('backend/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("✅ Vectorizer loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test with sample texts
    test_texts = [
        # Real news sample (should predict 0)
        "The United States government announced new economic policies today. Officials stated that the measures will help improve the country's financial stability and create more job opportunities for citizens.",
        
        # Fake news sample (should predict 1)
        "BREAKING: Scientists discover that drinking hot water with lemon cures all diseases instantly! This miracle cure has been hidden by big pharma for years. Share this immediately before they take it down!",
        
        # Another real news sample
        "NASA successfully launched a new satellite into orbit today. The satellite will help scientists study climate change and provide valuable data for environmental research.",
        
        # Another fake news sample
        "SHOCKING: Your phone is secretly recording everything you say! Tech companies are spying on you 24/7. Click here to learn the truth they don't want you to know!"
    ]
    
    print("\n🧪 Testing model predictions:")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        # Vectorize the text
        X = vectorizer.transform([text])
        
        # Get prediction
        prediction = model.predict(X)[0]
        
        # Get probability
        try:
            proba = model.predict_proba(X)[0]
            confidence = max(proba)
        except:
            confidence = 0.5
        
        # Determine expected result
        expected = "FAKE" if i in [2, 4] else "REAL"
        actual = "FAKE" if prediction == 1 else "REAL"
        
        print(f"\nTest {i}:")
        print(f"Text: {text[:100]}...")
        print(f"Expected: {expected}")
        print(f"Predicted: {actual} (value: {prediction})")
        print(f"Confidence: {confidence:.2%}")
        print(f"Status: {'✅' if expected == actual else '❌'}")
    
    # Test with actual dataset samples
    print("\n📊 Testing with actual dataset samples:")
    print("=" * 50)
    
    try:
        # Load a few samples from the dataset
        fake_df = pd.read_csv('data/Fake.csv')
        true_df = pd.read_csv('data/True.csv')
        
        # Test fake news samples
        for i in range(3):
            text = fake_df['text'].iloc[i]
            X = vectorizer.transform([text])
            prediction = model.predict(X)[0]
            actual = "FAKE" if prediction == 1 else "REAL"
            print(f"Fake sample {i+1}: Predicted {actual} (value: {prediction})")
        
        # Test real news samples
        for i in range(3):
            text = true_df['text'].iloc[i]
            X = vectorizer.transform([text])
            prediction = model.predict(X)[0]
            actual = "FAKE" if prediction == 1 else "REAL"
            print(f"Real sample {i+1}: Predicted {actual} (value: {prediction})")
            
    except Exception as e:
        print(f"❌ Error testing with dataset: {e}")

if __name__ == "__main__":
    test_model()
