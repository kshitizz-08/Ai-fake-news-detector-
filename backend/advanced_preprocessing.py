import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
import spacy
from collections import Counter
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class AdvancedTextPreprocessor:
    """
    Advanced text preprocessing with feature engineering
    for improved fake news detection
    """
    
    def __init__(self, use_spacy=True):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load spaCy model for advanced NLP features
        self.nlp = None
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ spaCy model loaded successfully")
            except OSError:
                print("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
        
        # Fake news indicators
        self.fake_indicators = {
            'clickbait_words': [
                'shocking', 'amazing', 'incredible', 'unbelievable', 'mind-blowing',
                'you won\'t believe', 'this will shock you', 'doctors hate this',
                'secret', 'exposed', 'revealed', 'conspiracy', 'cover-up'
            ],
            'emotional_words': [
                'outrageous', 'disgusting', 'terrifying', 'horrifying', 'shocking',
                'amazing', 'incredible', 'unbelievable', 'fantastic', 'wonderful'
            ],
            'urgency_words': [
                'breaking', 'urgent', 'immediate', 'now', 'today only',
                'limited time', 'act fast', 'don\'t wait', 'last chance'
            ],
            'authority_words': [
                'expert says', 'scientists confirm', 'doctors agree',
                'research shows', 'studies prove', 'official statement'
            ]
        }
        
        # Credibility indicators
        self.credibility_indicators = {
            'factual_words': [
                'according to', 'data shows', 'statistics indicate',
                'research conducted', 'study published', 'official report',
                'verified', 'confirmed', 'evidence', 'source'
            ],
            'neutral_words': [
                'reported', 'stated', 'announced', 'declared',
                'mentioned', 'noted', 'observed', 'found'
            ]
        }
    
    def preprocess_text(self, text, advanced_features=True):
        """
        Comprehensive text preprocessing with feature extraction
        """
        if not text or not isinstance(text, str):
            return "", {}
        
        # Basic preprocessing
        cleaned_text = self._basic_cleaning(text)
        
        # Advanced preprocessing
        if advanced_features:
            processed_text = self._advanced_cleaning(cleaned_text)
        else:
            processed_text = cleaned_text
        
        # Extract features
        features = {}
        if advanced_features:
            features = self._extract_text_features(text, processed_text)
        
        return processed_text, features
    
    def _basic_cleaning(self, text):
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'[\+]?[1-9][\d]{0,15}', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation (keep some for sentence structure)
        text = re.sub(r'[^\w\s\.\!\?]', '', text)
        
        return text.strip()
    
    def _advanced_cleaning(self, text):
        """Advanced text cleaning with NLP"""
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return ' '.join(tokens)
    
    def _extract_text_features(self, original_text, processed_text):
        """Extract comprehensive text features"""
        features = {}
        
        # Basic text statistics
        features.update(self._basic_statistics(original_text, processed_text))
        
        # Linguistic features
        features.update(self._linguistic_features(original_text))
        
        # Sentiment analysis
        features.update(self._sentiment_features(original_text))
        
        # Fake news indicators
        features.update(self._fake_news_indicators(original_text))
        
        # Credibility indicators
        features.update(self._credibility_indicators(original_text))
        
        # Readability metrics
        features.update(self._readability_features(original_text))
        
        # Advanced NLP features (if spaCy available)
        if self.nlp:
            features.update(self._spacy_features(original_text))
        
        return features
    
    def _basic_statistics(self, original_text, processed_text):
        """Extract basic text statistics"""
        # Word counts
        word_count = len(processed_text.split())
        char_count = len(original_text)
        sentence_count = len(sent_tokenize(original_text))
        
        # Vocabulary diversity
        unique_words = len(set(processed_text.split()))
        vocabulary_diversity = unique_words / max(word_count, 1)
        
        # Average word length
        avg_word_length = np.mean([len(word) for word in processed_text.split()]) if word_count > 0 else 0
        
        # Average sentence length
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'vocabulary_diversity': vocabulary_diversity,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'unique_words': unique_words
        }
    
    def _linguistic_features(self, text):
        """Extract linguistic features"""
        # Part of speech distribution
        tokens = word_tokenize(text.lower())
        
        # Count different types of words
        noun_count = 0
        verb_count = 0
        adj_count = 0
        adv_count = 0
        
        for token in tokens:
            if token in ['the', 'a', 'an', 'this', 'that', 'these', 'those']:
                noun_count += 1
            elif token.endswith(('ing', 'ed', 's')):
                verb_count += 1
            elif token.endswith(('ly', 'ward', 'wise')):
                adv_count += 1
            elif token.endswith(('al', 'ous', 'ful', 'less', 'able', 'ible')):
                adj_count += 1
        
        # Capitalization ratio
        capital_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Exclamation/question ratio
        exclamation_ratio = text.count('!') / max(len(text), 1)
        question_ratio = text.count('?') / max(len(text), 1)
        
        return {
            'noun_count': noun_count,
            'verb_count': verb_count,
            'adj_count': adj_count,
            'adv_count': adv_count,
            'capital_ratio': capital_ratio,
            'exclamation_ratio': exclamation_ratio,
            'question_ratio': question_ratio
        }
    
    def _sentiment_features(self, text):
        """Extract sentiment features"""
        blob = TextBlob(text)
        
        # Polarity and subjectivity
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Sentiment categories
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment_polarity': polarity,
            'sentiment_subjectivity': subjectivity,
            'sentiment_category': sentiment
        }
    
    def _fake_news_indicators(self, text):
        """Detect fake news indicators"""
        text_lower = text.lower()
        
        fake_scores = {}
        for category, words in self.fake_indicators.items():
            score = sum(1 for word in words if word.lower() in text_lower)
            fake_scores[f'{category}_count'] = score
        
        # Overall fake news score
        total_fake_score = sum(fake_scores.values())
        
        return {
            **fake_scores,
            'total_fake_indicators': total_fake_score,
            'fake_news_probability': min(total_fake_score / 10, 1.0)  # Normalize to 0-1
        }
    
    def _credibility_indicators(self, text):
        """Detect credibility indicators"""
        text_lower = text.lower()
        
        credibility_scores = {}
        for category, words in self.credibility_indicators.items():
            score = sum(1 for word in words if word.lower() in text_lower)
            credibility_scores[f'{category}_count'] = score
        
        # Overall credibility score
        total_credibility_score = sum(credibility_scores.values())
        
        return {
            **credibility_scores,
            'total_credibility_indicators': total_credibility_score,
            'credibility_probability': min(total_credibility_score / 10, 1.0)  # Normalize to 0-1
        }
    
    def _readability_features(self, text):
        """Calculate readability metrics"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Flesch Reading Ease
        if len(sentences) > 0 and len(words) > 0:
            avg_sentence_length = len(words) / len(sentences)
            syllables = self._count_syllables(text)
            avg_syllables_per_word = syllables / max(len(words), 1)
            
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            flesch_score = max(0, min(100, flesch_score))  # Clamp to 0-100
        else:
            flesch_score = 0
        
        # Reading level
        if flesch_score >= 90:
            reading_level = "Very Easy"
        elif flesch_score >= 80:
            reading_level = "Easy"
        elif flesch_score >= 70:
            reading_level = "Fairly Easy"
        elif flesch_score >= 60:
            reading_level = "Standard"
        elif flesch_score >= 50:
            reading_level = "Fairly Difficult"
        elif flesch_score >= 30:
            reading_level = "Difficult"
        else:
            reading_level = "Very Difficult"
        
        return {
            'flesch_reading_ease': flesch_score,
            'reading_level': reading_level,
            'avg_syllables_per_word': avg_syllables_per_word if 'avg_syllables_per_word' in locals() else 0
        }
    
    def _spacy_features(self, text):
        """Extract advanced NLP features using spaCy"""
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        
        # Named entities
        entity_types = Counter([ent.label_ for ent in doc.ents])
        
        # Dependency parsing features
        root_verbs = [token.text for token in doc if token.dep_ == 'ROOT']
        
        # Noun chunks
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        
        # Verb phrases
        verb_phrases = [token.text for token in doc if token.pos_ == 'VERB']
        
        return {
            'named_entities': dict(entity_types),
            'root_verbs': root_verbs,
            'noun_chunks_count': len(noun_chunks),
            'verb_phrases_count': len(verb_phrases),
            'spacy_available': True
        }
    
    def _count_syllables(self, text):
        """Simple syllable counting"""
        text = text.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        return count
    
    def create_feature_vector(self, features):
        """Convert features to numerical vector for ML models"""
        feature_vector = []
        feature_names = []
        
        # Basic statistics
        basic_stats = ['word_count', 'char_count', 'sentence_count', 'vocabulary_diversity', 
                      'avg_word_length', 'avg_sentence_length']
        
        for stat in basic_stats:
            if stat in features:
                feature_vector.append(features[stat])
                feature_names.append(stat)
        
        # Linguistic features
        linguistic_stats = ['noun_count', 'verb_count', 'adj_count', 'adv_count', 
                          'capital_ratio', 'exclamation_ratio', 'question_ratio']
        
        for stat in linguistic_stats:
            if stat in features:
                feature_vector.append(features[stat])
                feature_names.append(stat)
        
        # Sentiment features
        if 'sentiment_polarity' in features:
            feature_vector.append(features['sentiment_polarity'])
            feature_names.append('sentiment_polarity')
        
        if 'sentiment_subjectivity' in features:
            feature_vector.append(features['sentiment_subjectivity'])
            feature_names.append('sentiment_subjectivity')
        
        # Fake news indicators
        if 'total_fake_indicators' in features:
            feature_vector.append(features['total_fake_indicators'])
            feature_names.append('total_fake_indicators')
        
        # Credibility indicators
        if 'total_credibility_indicators' in features:
            feature_vector.append(features['total_credibility_indicators'])
            feature_names.append('total_credibility_indicators')
        
        # Readability
        if 'flesch_reading_ease' in features:
            feature_vector.append(features['flesch_reading_ease'])
            feature_names.append('flesch_reading_ease')
        
        return np.array(feature_vector), feature_names
    
    def get_feature_summary(self, features):
        """Get human-readable feature summary"""
        summary = {
            'text_quality': {
                'word_count': features.get('word_count', 0),
                'vocabulary_diversity': f"{features.get('vocabulary_diversity', 0):.2f}",
                'readability_level': features.get('reading_level', 'Unknown')
            },
            'sentiment_analysis': {
                'polarity': f"{features.get('sentiment_polarity', 0):.2f}",
                'subjectivity': f"{features.get('sentiment_subjectivity', 0):.2f}",
                'category': features.get('sentiment_category', 'Unknown')
            },
            'fake_news_risk': {
                'fake_indicators': features.get('total_fake_indicators', 0),
                'risk_level': self._calculate_risk_level(features.get('total_fake_indicators', 0)),
                'credibility_score': f"{features.get('credibility_probability', 0):.2f}"
            }
        }
        
        return summary
    
    def _calculate_risk_level(self, fake_indicators):
        """Calculate fake news risk level"""
        if fake_indicators == 0:
            return "Low Risk"
        elif fake_indicators <= 2:
            return "Medium Risk"
        elif fake_indicators <= 5:
            return "High Risk"
        else:
            return "Very High Risk"

# Example usage
if __name__ == "__main__":
    preprocessor = AdvancedTextPreprocessor()
    
    sample_text = """
    BREAKING NEWS: You won't BELIEVE what scientists just discovered! 
    This SHOCKING revelation will change everything you know about health.
    Doctors HATE this one simple trick that can cure any disease!
    """
    
    processed_text, features = preprocessor.preprocess_text(sample_text)
    
    print("🔍 Text Analysis Results:")
    print("=" * 50)
    print(f"Processed text: {processed_text[:100]}...")
    print("\n📊 Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    print("\n📋 Summary:")
    summary = preprocessor.get_feature_summary(features)
    for category, items in summary.items():
        print(f"\n{category.upper()}:")
        for key, value in items.items():
            print(f"  {key}: {value}")
