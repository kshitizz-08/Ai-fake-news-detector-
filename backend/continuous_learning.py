import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContinuousLearningSystem:
    """
    Continuous learning system that improves fake news detection models
    over time using user feedback and new data
    """
    
    def __init__(self, models_dir='models/', feedback_dir='feedback/'):
        self.models_dir = models_dir
        self.feedback_dir = feedback_dir
        self.feedback_file = os.path.join(feedback_dir, 'user_feedback.json')
        self.performance_history_file = os.path.join(feedback_dir, 'performance_history.json')
        self.model_versions_file = os.path.join(feedback_dir, 'model_versions.json')
        
        # Create directories if they don't exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Initialize feedback storage
        self.feedback_data = self._load_feedback_data()
        self.performance_history = self._load_performance_history()
        self.model_versions = self._load_model_versions()
        
        # Learning parameters
        self.min_feedback_threshold = 100  # Minimum feedback samples before retraining
        self.retraining_interval = 7  # Days between retraining attempts
        self.performance_threshold = 0.02  # Minimum improvement threshold for model update
        
        # Current model performance
        self.current_performance = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'last_updated': None
        }
        
        logger.info("🚀 Continuous Learning System initialized")
    
    def add_user_feedback(self, text, predicted_label, actual_label, user_id=None, confidence=None):
        """
        Add user feedback for model improvement
        
        Args:
            text: The news text that was analyzed
            predicted_label: What the model predicted (FAKE/REAL)
            actual_label: What the user believes is correct (FAKE/REAL)
            user_id: Optional user identifier
            confidence: Model confidence score
        """
        feedback_entry = {
            'id': self._generate_feedback_id(),
            'text': text,
            'predicted_label': predicted_label,
            'actual_label': actual_label,
            'user_id': user_id,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'text_length': len(text),
            'feedback_type': 'user_correction'
        }
        
        # Add to feedback data
        self.feedback_data.append(feedback_entry)
        
        # Save feedback
        self._save_feedback_data()
        
        logger.info(f"✅ User feedback added: {predicted_label} -> {actual_label}")
        
        # Check if retraining is needed
        if self._should_retrain():
            logger.info("🔄 Retraining threshold reached, initiating model update...")
            self._retrain_model()
    
    def add_batch_feedback(self, feedback_list):
        """Add multiple feedback entries at once"""
        for feedback in feedback_list:
            if isinstance(feedback, dict):
                self.add_user_feedback(**feedback)
            else:
                logger.warning(f"Invalid feedback format: {feedback}")
    
    def get_feedback_statistics(self):
        """Get statistics about collected feedback"""
        if not self.feedback_data:
            return {"total_feedback": 0}
        
        df = pd.DataFrame(self.feedback_data)
        
        # Basic statistics
        stats = {
            'total_feedback': len(df),
            'feedback_by_date': df.groupby(df['timestamp'].str[:10]).size().to_dict(),
            'feedback_by_type': df['feedback_type'].value_counts().to_dict(),
            'accuracy_discrepancies': len(df[df['predicted_label'] != df['actual_label']]),
            'confidence_analysis': {
                'high_confidence_errors': len(df[(df['confidence'] > 0.8) & (df['predicted_label'] != df['actual_label'])]),
                'low_confidence_errors': len(df[(df['confidence'] < 0.5) & (df['predicted_label'] != df['actual_label'])])
            }
        }
        
        # Performance over time
        if len(df) > 10:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily_accuracy = df.groupby('date').apply(
                lambda x: (x['predicted_label'] == x['actual_label']).mean()
            )
            stats['daily_accuracy'] = daily_accuracy.to_dict()
        
        return stats
    
    def _should_retrain(self):
        """Determine if model retraining is needed"""
        # Check feedback threshold
        if len(self.feedback_data) < self.min_feedback_threshold:
            return False
        
        # Check time interval
        if self.model_versions:
            last_training = max(self.model_versions.keys())
            last_training_date = datetime.fromisoformat(last_training)
            days_since_training = (datetime.now() - last_training_date).days
            
            if days_since_training < self.retraining_interval:
                return False
        
        # Check if there are enough new feedback samples
        if self.model_versions:
            last_training = max(self.model_versions.keys())
            new_feedback_count = len([
                f for f in self.feedback_data 
                if f['timestamp'] > last_training
            ])
            
            if new_feedback_count < self.min_feedback_threshold // 2:
                return False
        
        return True
    
    def _retrain_model(self):
        """Retrain the model using collected feedback"""
        try:
            logger.info("🔄 Starting model retraining...")
            
            # Prepare training data from feedback
            X_train, y_train = self._prepare_training_data()
            
            if len(X_train) < self.min_feedback_threshold:
                logger.warning(f"Insufficient training data: {len(X_train)} samples")
                return False
            
            # Split data
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Train new model
            new_model = self._train_new_model(X_train_split, y_train_split)
            
            # Evaluate new model
            new_performance = self._evaluate_model(new_model, X_val_split, y_val_split)
            
            # Compare with current performance
            if self._should_update_model(new_performance):
                # Save new model
                self._save_new_model(new_model, new_performance)
                
                # Update performance history
                self._update_performance_history(new_performance)
                
                logger.info("✅ Model successfully updated with new performance")
                return True
            else:
                logger.info("⚠️ New model performance not sufficient for update")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error during model retraining: {e}")
            return False
    
    def _prepare_training_data(self):
        """Prepare training data from feedback"""
        # Convert feedback to training format
        texts = [f['text'] for f in self.feedback_data]
        labels = [1 if f['actual_label'] == 'FAKE' else 0 for f in self.feedback_data]
        
        # Basic text preprocessing (you can integrate with AdvancedTextPreprocessor)
        processed_texts = self._basic_text_preprocessing(texts)
        
        # Create TF-IDF features
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        if not hasattr(self, 'vectorizer'):
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            self.vectorizer.fit(processed_texts)
        
        X = self.vectorizer.transform(processed_texts)
        y = np.array(labels)
        
        return X, y
    
    def _basic_text_preprocessing(self, texts):
        """Basic text preprocessing for training"""
        processed = []
        for text in texts:
            # Convert to lowercase
            text = text.lower()
            # Remove special characters
            text = ''.join(c for c in text if c.isalnum() or c.isspace())
            # Remove extra whitespace
            text = ' '.join(text.split())
            processed.append(text)
        return processed
    
    def _train_new_model(self, X_train, y_train):
        """Train a new model using the feedback data"""
        # Use ensemble approach
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        # Create base models
        models = [
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]
        
        # Create ensemble
        ensemble = VotingClassifier(estimators=models, voting='soft')
        
        # Train
        ensemble.fit(X_train, y_train)
        
        return ensemble
    
    def _evaluate_model(self, model, X_val, y_val):
        """Evaluate model performance"""
        y_pred = model.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'timestamp': datetime.now().isoformat()
        }
    
    def _should_update_model(self, new_performance):
        """Determine if new model should replace current model"""
        if not self.current_performance['accuracy']:
            return True  # First model
        
        # Check if improvement exceeds threshold
        accuracy_improvement = new_performance['accuracy'] - self.current_performance['accuracy']
        f1_improvement = new_performance['f1_score'] - self.current_performance['f1_score']
        
        return (accuracy_improvement > self.performance_threshold or 
                f1_improvement > self.performance_threshold)
    
    def _save_new_model(self, model, performance):
        """Save the new model and update version tracking"""
        # Generate version ID
        version_id = datetime.now().isoformat()
        
        # Save model
        model_path = os.path.join(self.models_dir, f'model_v{version_id}.pkl')
        joblib.dump(model, model_path)
        
        # Save vectorizer
        vectorizer_path = os.path.join(self.models_dir, f'vectorizer_v{version_id}.pkl')
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Update version tracking
        self.model_versions[version_id] = {
            'model_path': model_path,
            'vectorizer_path': vectorizer_path,
            'performance': performance,
            'feedback_count': len(self.feedback_data),
            'training_date': version_id
        }
        
        # Update current performance
        self.current_performance = performance
        
        # Save version info
        self._save_model_versions()
        
        logger.info(f"✅ New model saved as version {version_id}")
    
    def _update_performance_history(self, performance):
        """Update performance history"""
        timestamp = performance['timestamp']
        self.performance_history[timestamp] = performance
        self._save_performance_history()
    
    def get_model_performance_trends(self):
        """Analyze model performance trends over time"""
        if not self.performance_history:
            return {"message": "No performance history available"}
        
        df = pd.DataFrame.from_dict(self.performance_history, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        trends = {
            'overall_trend': {
                'accuracy': df['accuracy'].tolist(),
                'f1_score': df['f1_score'].tolist(),
                'timestamps': df.index.strftime('%Y-%m-%d').tolist()
            },
            'improvement_rate': {
                'accuracy_improvement': (df['accuracy'].iloc[-1] - df['accuracy'].iloc[0]) / len(df),
                'f1_improvement': (df['f1_score'].iloc[-1] - df['f1_score'].iloc[0]) / len(df)
            },
            'stability': {
                'accuracy_std': df['accuracy'].std(),
                'f1_std': df['f1_score'].std()
            }
        }
        
        return trends
    
    def _generate_feedback_id(self):
        """Generate unique feedback ID"""
        return f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.feedback_data)}"
    
    def _load_feedback_data(self):
        """Load feedback data from file"""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")
        return []
    
    def _save_feedback_data(self):
        """Save feedback data to file"""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving feedback data: {e}")
    
    def _load_performance_history(self):
        """Load performance history from file"""
        try:
            if os.path.exists(self.performance_history_file):
                with open(self.performance_history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading performance history: {e}")
        return {}
    
    def _save_performance_history(self):
        """Save performance history to file"""
        try:
            with open(self.performance_history_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance history: {e}")
    
    def _load_model_versions(self):
        """Load model version information from file"""
        try:
            if os.path.exists(self.model_versions_file):
                with open(self.model_versions_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading model versions: {e}")
        return {}
    
    def _save_model_versions(self):
        """Save model version information to file"""
        try:
            with open(self.model_versions_file, 'w') as f:
                json.dump(self.model_versions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model versions: {e}")
    
    def get_system_status(self):
        """Get comprehensive system status"""
        return {
            'feedback_collected': len(self.feedback_data),
            'model_versions': len(self.model_versions),
            'current_performance': self.current_performance,
            'retraining_threshold': self.min_feedback_threshold,
            'days_since_last_training': self._get_days_since_training(),
            'feedback_statistics': self.get_feedback_statistics(),
            'performance_trends': self.get_model_performance_trends()
        }
    
    def _get_days_since_training(self):
        """Get days since last model training"""
        if not self.model_versions:
            return None
        
        last_training = max(self.model_versions.keys())
        last_training_date = datetime.fromisoformat(last_training)
        return (datetime.now() - last_training_date).days

# Example usage
if __name__ == "__main__":
    # Initialize continuous learning system
    cl_system = ContinuousLearningSystem()
    
    # Add some sample feedback
    sample_feedback = [
        {
            'text': 'BREAKING: Scientists discover amazing cure for all diseases!',
            'predicted_label': 'FAKE',
            'actual_label': 'FAKE',
            'user_id': 'user1',
            'confidence': 0.95
        },
        {
            'text': 'New study shows moderate exercise improves heart health',
            'predicted_label': 'REAL',
            'actual_label': 'REAL',
            'user_id': 'user2',
            'confidence': 0.87
        }
    ]
    
    # Add feedback
    for feedback in sample_feedback:
        cl_system.add_user_feedback(**feedback)
    
    # Get system status
    status = cl_system.get_system_status()
    print("🔍 Continuous Learning System Status:")
    print(json.dumps(status, indent=2))
