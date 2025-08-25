import pickle
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class EnsembleFakeNewsDetector:
    """
    Ensemble model system that combines multiple ML algorithms
    for improved fake news detection accuracy
    """
    
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.ensemble_model = None
        self.feature_importance = {}
        self.model_performance = {}
        
    def create_models(self):
        """Create individual base models"""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                C=1.0
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            ),
            'svm': SVC(
                kernel='linear',
                random_state=42,
                probability=True
            ),
            'naive_bayes': MultinomialNB(
                alpha=1.0
            )
        }
        
        print("✅ Base models created successfully")
        
    def create_ensemble(self, voting_method='soft'):
        """Create ensemble model using voting classifier"""
        estimators = [(name, model) for name, model in self.models.items()]
        
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting=voting_method,
            n_jobs=-1
        )
        
        print(f"✅ Ensemble model created with {voting_method} voting")
        
    def train_ensemble(self, X_train, y_train, X_val=None, y_val=None):
        """Train the ensemble model"""
        print("🚀 Training ensemble model...")
        
        # Train individual models first
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Calculate individual model performance
            if X_val is not None and y_val is not None:
                y_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                self.model_performance[name] = accuracy
                print(f"{name} accuracy: {accuracy:.4f}")
        
        # Train ensemble model
        print("Training ensemble...")
        self.ensemble_model.fit(X_train, y_train)
        
        # Ensemble performance
        if X_val is not None and y_val is not None:
            y_pred_ensemble = self.ensemble_model.predict(X_val)
            ensemble_accuracy = accuracy_score(y_val, y_pred_ensemble)
            self.model_performance['ensemble'] = ensemble_accuracy
            print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        
        print("✅ Ensemble training completed!")
        
    def predict_with_confidence(self, text, vectorizer):
        """Make prediction with confidence scores and model agreement"""
        # Vectorize text
        X = vectorizer.transform([text])
        
        # Get predictions from all models
        predictions = {}
        confidences = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                pred = model.predict(X)[0]
                confidence = max(proba)
                predictions[name] = pred
                confidences[name] = confidence
        
        # Ensemble prediction
        ensemble_pred = self.ensemble_model.predict(X)[0]
        if hasattr(self.ensemble_model, 'predict_proba'):
            ensemble_proba = self.ensemble_model.predict_proba(X)[0]
            ensemble_confidence = max(ensemble_proba)
        else:
            ensemble_confidence = 0.5
        
        # Calculate model agreement
        pred_values = list(predictions.values())
        agreement_ratio = pred_values.count(ensemble_pred) / len(pred_values)
        
        # Feature importance analysis
        feature_importance = self._analyze_feature_importance(X, vectorizer)
        
        return {
            'prediction': 'FAKE' if ensemble_pred == 1 else 'REAL',
            'confidence': ensemble_confidence,
            'model_agreement': agreement_ratio,
            'individual_predictions': predictions,
            'individual_confidences': confidences,
            'feature_importance': feature_importance,
            'ensemble_confidence': ensemble_confidence
        }
    
    def _analyze_feature_importance(self, X, vectorizer):
        """Analyze feature importance across all models"""
        feature_names = vectorizer.get_feature_names_out()
        importance_scores = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_[0])
            else:
                continue
                
            # Get top features
            top_indices = np.argsort(importances)[-10:][::-1]
            top_features = [(feature_names[i], importances[i]) for i in top_indices]
            importance_scores[name] = top_features
        
        return importance_scores
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='logistic_regression'):
        """Perform hyperparameter tuning for specific models"""
        print(f"🔧 Tuning hyperparameters for {model_name}...")
        
        if model_name == 'logistic_regression':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        elif model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        elif model_name == 'svm':
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        else:
            print(f"❌ Hyperparameter tuning not supported for {model_name}")
            return
        
        # Grid search
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        print(f"✅ Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
    def cross_validate_models(self, X, y, cv=5):
        """Perform cross-validation for all models"""
        print("🔍 Performing cross-validation...")
        
        cv_scores = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            cv_scores[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # Ensemble cross-validation
        ensemble_scores = cross_val_score(self.ensemble_model, X, y, cv=cv, scoring='accuracy')
        cv_scores['ensemble'] = {
            'mean': ensemble_scores.mean(),
            'std': ensemble_scores.std(),
            'scores': ensemble_scores
        }
        print(f"Ensemble: {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def save_models(self, filepath='models/'):
        """Save all trained models"""
        os.makedirs(filepath, exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            model_path = os.path.join(filepath, f'{name}.pkl')
            joblib.dump(model, model_path)
            print(f"✅ Saved {name} to {model_path}")
        
        # Save ensemble model
        ensemble_path = os.path.join(filepath, 'ensemble_model.pkl')
        joblib.dump(self.ensemble_model, ensemble_path)
        print(f"✅ Saved ensemble model to {ensemble_path}")
        
        # Save performance metrics
        metrics_path = os.path.join(filepath, 'model_performance.json')
        import json
        with open(metrics_path, 'w') as f:
            json.dump(self.model_performance, f, indent=2)
        print(f"✅ Saved performance metrics to {metrics_path}")
    
    def load_models(self, filepath='models/'):
        """Load pre-trained models"""
        try:
            # Load individual models
            for name in self.models.keys():
                model_path = os.path.join(filepath, f'{name}.pkl')
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
                    print(f"✅ Loaded {name} from {model_path}")
            
            # Load ensemble model
            ensemble_path = os.path.join(filepath, 'ensemble_model.pkl')
            if os.path.exists(ensemble_path):
                self.ensemble_model = joblib.load(ensemble_path)
                print(f"✅ Loaded ensemble model from {ensemble_path}")
            
            # Load performance metrics
            metrics_path = os.path.join(filepath, 'model_performance.json')
            if os.path.exists(metrics_path):
                import json
                with open(metrics_path, 'r') as f:
                    self.model_performance = json.load(f)
                print(f"✅ Loaded performance metrics from {metrics_path}")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def get_model_summary(self):
        """Get comprehensive model performance summary"""
        summary = {
            'total_models': len(self.models),
            'ensemble_available': self.ensemble_model is not None,
            'individual_performance': self.model_performance,
            'best_individual_model': None,
            'ensemble_performance': None
        }
        
        if self.model_performance:
            # Find best individual model
            individual_scores = {k: v for k, v in self.model_performance.items() 
                               if k != 'ensemble'}
            if individual_scores:
                best_model = max(individual_scores.items(), key=lambda x: x[1])
                summary['best_individual_model'] = {
                    'name': best_model[0],
                    'accuracy': best_model[1]
                }
            
            # Ensemble performance
            if 'ensemble' in self.model_performance:
                summary['ensemble_performance'] = self.model_performance['ensemble']
        
        return summary

# Example usage
if __name__ == "__main__":
    # Initialize ensemble detector
    detector = EnsembleFakeNewsDetector()
    
    # Create models
    detector.create_models()
    
    # Create ensemble
    detector.create_ensemble(voting_method='soft')
    
    print("🎯 Ensemble Fake News Detector initialized successfully!")
    print("Use detector.load_models() to load pre-trained models")
    print("Use detector.train_ensemble() to train new models")
