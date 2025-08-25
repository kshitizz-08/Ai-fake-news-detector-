#!/usr/bin/env python3
"""
Enhanced Fake News Detection Model Training Script
Integrates ensemble models, advanced preprocessing, and continuous learning
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add backend to path
sys.path.append('backend')

from ensemble_model import EnsembleFakeNewsDetector
from advanced_preprocessing import AdvancedTextPreprocessor
from continuous_learning import ContinuousLearningSystem

class EnhancedModelTrainer:
    """
    Enhanced model trainer that integrates all ML improvements
    """
    
    def __init__(self, data_dir='data/', models_dir='backend/models/'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.preprocessor = AdvancedTextPreprocessor()
        self.ensemble_detector = EnsembleFakeNewsDetector()
        self.continuous_learning = ContinuousLearningSystem()
        
        # Create directories
        os.makedirs(models_dir, exist_ok=True)
        
        print("🚀 Enhanced Model Trainer initialized")
    
    def load_and_prepare_data(self):
        """Load and prepare training data with advanced preprocessing"""
        print("📊 Loading training data...")
        
        # Load datasets
        try:
            fake_df = pd.read_csv(os.path.join(self.data_dir, 'Fake.csv'))
            true_df = pd.read_csv(os.path.join(self.data_dir, 'True.csv'))
            
            print(f"✅ Loaded {len(fake_df)} fake news samples")
            print(f"✅ Loaded {len(true_df)} true news samples")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            return None, None
        
        # Prepare labels
        fake_df['label'] = 1  # 1 for FAKE
        true_df['label'] = 0  # 0 for REAL
        
        # Combine datasets
        combined_df = pd.concat([fake_df, true_df], ignore_index=True)
        
        # Shuffle data
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✅ Combined dataset: {len(combined_df)} total samples")
        
        return combined_df
    
    def advanced_preprocessing_pipeline(self, texts, labels):
        """Advanced preprocessing pipeline with feature engineering"""
        print("🔧 Running advanced preprocessing pipeline...")
        
        processed_texts = []
        all_features = []
        
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f"Processing text {i+1}/{len(texts)}...")
            
            # Advanced preprocessing
            processed_text, features = self.preprocessor.preprocess_text(text, advanced_features=True)
            processed_texts.append(processed_text)
            all_features.append(features)
        
        print("✅ Advanced preprocessing completed")
        
        # Create feature matrix
        feature_vectors = []
        for features in all_features:
            vector, _ = self.preprocessor.create_feature_vector(features)
            feature_vectors.append(vector)
        
        feature_matrix = np.array(feature_vectors)
        
        return processed_texts, feature_matrix, all_features
    
    def create_enhanced_features(self, texts, basic_features):
        """Create enhanced feature set combining TF-IDF and engineered features"""
        print("🔧 Creating enhanced feature set...")
        
        # TF-IDF features
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        tfidf_features = tfidf_vectorizer.fit_transform(texts)
        print(f"✅ TF-IDF features: {tfidf_features.shape}")
        
        # Combine TF-IDF with engineered features
        if basic_features is not None and len(basic_features) > 0:
            # Ensure feature matrix has correct shape
            if basic_features.shape[1] > 0:
                from scipy.sparse import hstack
                enhanced_features = hstack([tfidf_features, basic_features])
                print(f"✅ Enhanced features: {enhanced_features.shape}")
            else:
                enhanced_features = tfidf_features
                print("⚠️ No engineered features available, using TF-IDF only")
        else:
            enhanced_features = tfidf_features
            print("⚠️ No engineered features available, using TF-IDF only")
        
        return enhanced_features, tfidf_vectorizer
    
    def train_ensemble_model(self, X_train, y_train, X_val, y_val):
        """Train ensemble model with hyperparameter tuning"""
        print("🎯 Training enhanced ensemble model...")
        
        # Create and configure ensemble detector
        self.ensemble_detector.create_models()
        self.ensemble_detector.create_ensemble(voting_method='soft')
        
        # Hyperparameter tuning for key models
        print("🔧 Performing hyperparameter tuning...")
        
        # Tune Logistic Regression
        self.ensemble_detector.hyperparameter_tuning(X_train, y_train, 'logistic_regression')
        
        # Tune Random Forest
        self.ensemble_detector.hyperparameter_tuning(X_train, y_train, 'random_forest')
        
        # Train ensemble with tuned models
        self.ensemble_detector.train_ensemble(X_train, y_train, X_val, y_val)
        
        print("✅ Ensemble model training completed")
        
        return self.ensemble_detector
    
    def evaluate_model_comprehensive(self, model, X_test, y_test, X_val, y_val):
        """Comprehensive model evaluation"""
        print("📊 Comprehensive model evaluation...")
        
        # Test set predictions
        y_pred_test = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Validation set predictions
        y_pred_val = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_pred_val)
        
        # Detailed metrics
        test_report = classification_report(y_test, y_pred_test, target_names=['REAL', 'FAKE'])
        test_confusion = confusion_matrix(y_test, y_pred_test)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
        
        evaluation_results = {
            'test_accuracy': test_accuracy,
            'validation_accuracy': val_accuracy,
            'cross_validation_mean': cv_scores.mean(),
            'cross_validation_std': cv_scores.std(),
            'classification_report': test_report,
            'confusion_matrix': test_confusion.tolist(),
            'cv_scores': cv_scores.tolist()
        }
        
        print("📊 Evaluation Results:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Cross-Validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return evaluation_results
    
    def save_enhanced_model(self, model, vectorizer, evaluation_results):
        """Save enhanced model with all components"""
        print("💾 Saving enhanced model...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save ensemble model
        ensemble_path = os.path.join(self.models_dir, f'enhanced_ensemble_{timestamp}.pkl')
        joblib.dump(model, ensemble_path)
        
        # Save TF-IDF vectorizer
        vectorizer_path = os.path.join(self.models_dir, f'enhanced_vectorizer_{timestamp}.pkl')
        joblib.dump(vectorizer, vectorizer_path)
        
        # Save evaluation results
        results_path = os.path.join(self.models_dir, f'evaluation_results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Save model metadata
        metadata = {
            'timestamp': timestamp,
            'model_type': 'Enhanced Ensemble',
            'training_date': datetime.now().isoformat(),
            'evaluation_results': evaluation_results,
            'model_path': ensemble_path,
            'vectorizer_path': vectorizer_path,
            'results_path': results_path
        }
        
        metadata_path = os.path.join(self.models_dir, f'model_metadata_{timestamp}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Enhanced model saved with timestamp: {timestamp}")
        print(f"📁 Model files saved in: {self.models_dir}")
        
        return timestamp
    
    def run_complete_training_pipeline(self):
        """Run complete enhanced training pipeline"""
        print("🚀 Starting Enhanced Training Pipeline")
        print("=" * 60)
        
        # Step 1: Load and prepare data
        combined_df = self.load_and_prepare_data()
        if combined_df is None:
            return False
        
        # Step 2: Split data
        print("📊 Splitting data into train/validation/test sets...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            combined_df['text'], combined_df['label'], 
            test_size=0.2, random_state=42, stratify=combined_df['label']
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        print(f"✅ Training set: {len(X_train)} samples")
        print(f"✅ Validation set: {len(X_val)} samples")
        print(f"✅ Test set: {len(X_test)} samples")
        
        # Step 3: Advanced preprocessing
        processed_train, train_features, _ = self.advanced_preprocessing_pipeline(X_train, y_train)
        processed_val, val_features, _ = self.advanced_preprocessing_pipeline(X_val, y_val)
        processed_test, test_features, _ = self.advanced_preprocessing_pipeline(X_test, y_test)
        
        # Step 4: Create enhanced features
        X_train_enhanced, tfidf_vectorizer = self.create_enhanced_features(processed_train, train_features)
        X_val_enhanced, _ = self.create_enhanced_features(processed_val, val_features)
        X_test_enhanced, _ = self.create_enhanced_features(processed_test, test_features)
        
        # Step 5: Train ensemble model
        ensemble_model = self.train_ensemble_model(X_train_enhanced, y_train, X_val_enhanced, y_val)
        
        # Step 6: Evaluate model
        evaluation_results = self.evaluate_model_comprehensive(
            ensemble_model, X_test_enhanced, y_test, X_val_enhanced, y_val
        )
        
        # Step 7: Save enhanced model
        timestamp = self.save_enhanced_model(ensemble_model, tfidf_vectorizer, evaluation_results)
        
        # Step 8: Initialize continuous learning
        print("🔄 Initializing continuous learning system...")
        self.continuous_learning.current_performance = {
            'accuracy': evaluation_results['test_accuracy'],
            'precision': float(evaluation_results['classification_report'].split('\n')[2].split()[1]),
            'recall': float(evaluation_results['classification_report'].split('\n')[2].split()[2]),
            'f1_score': float(evaluation_results['classification_report'].split('\n')[2].split()[3]),
            'last_updated': datetime.now().isoformat()
        }
        
        print("✅ Enhanced Training Pipeline completed successfully!")
        print(f"📊 Final Model Performance: {evaluation_results['test_accuracy']:.4f}")
        
        return True
    
    def generate_training_report(self):
        """Generate comprehensive training report"""
        print("📋 Generating training report...")
        
        report = {
            'training_summary': {
                'date': datetime.now().isoformat(),
                'model_type': 'Enhanced Ensemble',
                'data_sources': ['Fake.csv', 'True.csv'],
                'preprocessing': 'Advanced NLP Pipeline',
                'feature_engineering': 'TF-IDF + Engineered Features',
                'model_architecture': 'Voting Classifier Ensemble'
            },
            'continuous_learning': {
                'enabled': True,
                'feedback_threshold': self.continuous_learning.min_feedback_threshold,
                'retraining_interval': self.continuous_learning.retraining_interval
            },
            'next_steps': [
                'Model is ready for production use',
                'Continuous learning system is active',
                'Monitor performance and collect user feedback',
                'Automatic retraining will occur based on feedback'
            ]
        }
        
        # Save report
        report_path = os.path.join(self.models_dir, 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print("✅ Training report generated and saved")
        return report

def main():
    """Main training function"""
    print("🎯 Enhanced Fake News Detection Model Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = EnhancedModelTrainer()
    
    # Run training pipeline
    success = trainer.run_complete_training_pipeline()
    
    if success:
        # Generate report
        report = trainer.generate_training_report()
        
        print("\n🎉 Training completed successfully!")
        print("=" * 60)
        print("📋 Summary:")
        print("✅ Enhanced ensemble model trained")
        print("✅ Advanced preprocessing pipeline implemented")
        print("✅ Continuous learning system activated")
        print("✅ Model ready for production use")
        
        print("\n🚀 Next steps:")
        print("1. Start the backend server: python backend/app.py")
        print("2. Open the frontend: frontend/index.html")
        print("3. Test the enhanced model with news articles")
        print("4. Monitor performance and collect user feedback")
        
    else:
        print("❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
