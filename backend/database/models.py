# backend/database/models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='user')  # user, fact_checker, publisher, admin
    reputation_score = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    feedbacks = db.relationship('Feedback', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Feedback(db.Model):
    __tablename__ = 'feedbacks'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    article_text = db.Column(db.Text, nullable=False)
    predicted_label = db.Column(db.String(10), nullable=False)
    correct_label = db.Column(db.String(10), nullable=False)
    language = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float)
    validated = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AnalysisLog(db.Model):
    __tablename__ = 'analysis_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    content_hash = db.Column(db.String(64), unique=True)
    language = db.Column(db.String(20))
    prediction = db.Column(db.String(10))
    confidence = db.Column(db.Float)
    analyzed_at = db.Column(db.DateTime, default=datetime.utcnow)