#!/usr/bin/env python3
"""
Debug script to test the top_features functionality
"""

import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def test_top_features():
    # Load model and vectorizer
    model_path = os.path.join("backend", "model.pkl")
    vectorizer_path = os.path.join("backend", "vectorizer.pkl")
    
    print("Loading model and vectorizer...")
    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
    
    print(f"Model type: {type(model)}")
    print(f"Vectorizer type: {type(vectorizer)}")
    
    # Test text
    test_text = "This is a test news article about politics and elections. The government announced new policies today."
    
    # Clean text (simplified version)
    cleaned_text = test_text.lower()
    
    # Transform text
    X = vectorizer.transform([cleaned_text])
    print(f"Feature matrix shape: {X.shape}")
    
    # Get prediction
    prediction = model.predict(X)[0]
    print(f"Prediction: {prediction}")
    
    # Get feature importance for Naive Bayes
    feature_names = vectorizer.get_feature_names_out()
    print(f"Number of features: {len(feature_names)}")
    
    if hasattr(model, 'feature_log_prob_'):
        print("Using Naive Bayes feature importance...")
        # Use the difference between log probabilities of classes
        if len(model.classes_) == 2:
            # For binary classification, use the difference between fake and real class probabilities
            fake_class_idx = 1 if 1 in model.classes_ else 0
            real_class_idx = 0 if fake_class_idx == 1 else 1
            feature_importance = model.feature_log_prob_[fake_class_idx] - model.feature_log_prob_[real_class_idx]
            print(f"Fake class index: {fake_class_idx}, Real class index: {real_class_idx}")
        else:
            feature_importance = model.feature_log_prob_[0]  # Use first class as reference
        
        print(f"Feature importance shape: {feature_importance.shape}")
        
        # Get input features
        input_features = X.toarray()[0]
        non_zero_indices = [i for i, val in enumerate(input_features) if val > 0]
        print(f"Non-zero features in input: {len(non_zero_indices)}")
        
        # Calculate contribution scores
        feature_scores = []
        for idx in non_zero_indices:
            if idx < len(feature_importance):
                score = input_features[idx] * feature_importance[idx]
                feature_scores.append((feature_names[idx], score, input_features[idx]))
        
        # Sort by absolute contribution score
        feature_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = feature_scores[:10]
        
        print(f"\nTop 10 contributing features:")
        for i, (word, score, freq) in enumerate(top_features):
            print(f"{i+1}. {word}: score={score:.4f}, freq={freq:.4f}")
    else:
        print("Model doesn't have feature_log_prob_ attribute")
        print(f"Model attributes: {dir(model)}")

if __name__ == "__main__":
    test_top_features()
