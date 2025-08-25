#!/usr/bin/env python3
"""
Database initialization script for Fake News Detection app
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # User activity tracking
    predictions_made = db.Column(db.Integer, default=0)
    fake_detected = db.Column(db.Integer, default=0)
    real_detected = db.Column(db.Integer, default=0)

# News model for storing analyzed articles
class News(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(500), nullable=False)
    content = db.Column(db.Text, nullable=False)
    cleaned_content = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(10), nullable=False)  # FAKE or REAL
    confidence = db.Column(db.Float, nullable=False)
    source_type = db.Column(db.String(20), nullable=False)  # text or url
    original_source = db.Column(db.String(1000))  # URL if applicable
    analyzed_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    # Text analysis features
    word_count = db.Column(db.Integer)
    vocabulary_diversity = db.Column(db.Float)
    readability_score = db.Column(db.Float)
    formal_indicators = db.Column(db.Integer)
    credibility_indicators = db.Column(db.Integer)
    emotional_indicators = db.Column(db.Integer)
    
    # Vector representation for similarity search
    content_vector = db.Column(db.Text)  # JSON string of TF-IDF vector
    
    # Related news tracking
    related_news_ids = db.Column(db.Text)  # JSON string of related article IDs
    
    user = db.relationship('User', backref='news_articles')

def init_database():
    with app.app_context():
        # Create all tables
        db.create_all()
        
        print("Database initialized successfully!")
        print("Tables created:")
        print("- users")
        print("- news")
        
        # Check if tables exist
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        print(f"Available tables: {tables}")

if __name__ == '__main__':
    init_database()
