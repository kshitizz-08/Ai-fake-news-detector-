<<<<<<< HEAD
# 🔍 Enhanced Fake News Detection System

A comprehensive web application that detects fake news using advanced NLP techniques and connects analyzed content to previously related information for deeper insights.

## ✨ Key Features

### 🎯 **Core News Detection**
- **AI-Powered Analysis**: Uses machine learning models to classify news as real or fake
- **URL & Text Support**: Analyze news from URLs or direct text input
- **Confidence Scoring**: Provides confidence levels for each prediction
- **Real-time Processing**: Instant analysis with detailed explanations

### 🔗 **Related News Detection**
- **Content Similarity**: Automatically finds previously stored news with similar content
- **Pattern Recognition**: Identifies common themes and language patterns
- **Cross-Reference Analysis**: Compare current news with historical data
- **Similarity Scoring**: Percentage-based similarity matching



### 📚 **User History & Analytics**
- **Personal Dashboard**: Track your analysis history and statistics
- **Progress Monitoring**: See your fake news detection patterns
- **Performance Metrics**: Understand your analysis accuracy over time
- **Pagination Support**: Navigate through extensive history efficiently



### 🎨 **Modern User Interface**
- **Responsive Design**: Works seamlessly on all devices
- **Beautiful Gradients**: Modern, visually appealing interface
- **Interactive Elements**: Hover effects and smooth animations
- **Accessibility Features**: Screen reader support and keyboard navigation

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Fake-News-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the database**
   ```bash
   python init_db.py
   ```

4. **Start the backend server**
   ```bash
   cd backend
   python app.py
   ```

5. **Open the frontend**
   - Navigate to `frontend/index.html` in your browser
   - Or use a local server: `python -m http.server 8000`

## 🏗️ Architecture

### Backend (Flask)
- **Flask Web Framework**: RESTful API endpoints
- **SQLAlchemy ORM**: Database management and models
- **Machine Learning**: Pre-trained models for news classification
- **Vector Similarity**: TF-IDF based content matching
- **User Authentication**: Secure login and session management

### Frontend (HTML/CSS/JavaScript)
- **Vanilla JavaScript**: No framework dependencies
- **Modern CSS**: Flexbox, Grid, and CSS animations
- **Responsive Design**: Mobile-first approach
- **Progressive Enhancement**: Works without JavaScript

### Database
- **SQLite**: Lightweight, file-based database
- **User Management**: Authentication and user profiles
- **News Storage**: Analyzed articles with metadata
- **Vector Storage**: Content vectors for similarity search

## 🔧 API Endpoints

### Authentication
- `POST /register` - User registration
- `POST /login` - User login
- `POST /logout` - User logout
- `GET /check-auth` - Check authentication status

### News Analysis
- `POST /predict` - Analyze news content
- `GET /news/<id>` - Get news article details
- `GET /news/history` - Get user's analysis history

### User Management
- `GET /user/profile` - Get user profile
- `GET /user/stats` - Get user statistics

## 📊 How It Works

### 1. **Content Analysis**
- Input news text or URL
- Text preprocessing and cleaning
- Feature extraction and vectorization
- ML model prediction with confidence scoring



### 3. **Pattern Recognition**
- Analyze text characteristics
- Identify credibility indicators
- Track emotional language patterns
- Monitor formal vs. informal writing



## 🎯 Use Cases

### **For Individuals**
- Verify news articles before sharing
- Learn to identify fake news patterns
- Track personal analysis history
- Build media literacy skills

### **For Researchers**
- Analyze misinformation trends
- Study language patterns in fake news
- Track temporal evolution of fake news
- Generate datasets for research

### **For Educators**
- Teach media literacy
- Demonstrate fake news detection
- Show real-world examples
- Interactive learning tool

## 🔒 Security Features

- **Password Hashing**: Secure password storage
- **Session Management**: Secure user sessions
- **Input Validation**: Sanitized user inputs
- **CORS Protection**: Cross-origin request handling
- **SQL Injection Prevention**: Parameterized queries

## 📱 Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Machine learning models trained on public datasets
- Open-source libraries and frameworks
- Research on fake news detection techniques
- Community contributions and feedback

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation

---

**Built with ❤️ for a more informed and media-literate world**

=======
# Ai-fake-news-detector-
AI-powered Fake News Detection system using NLP and Machine Learning to classify news as real or fake. Features include text/URL input, voice support, real-time detection, and advanced analytics. Designed to help users, journalists, and researchers verify news credibility and combat misinformation.
>>>>>>> 084ed14f9e7136458fbe675a7b6b86d44498638c
