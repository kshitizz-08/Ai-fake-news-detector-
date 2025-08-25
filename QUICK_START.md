# 🚀 Quick Start Guide - Fake News Detection Project

## 📋 Prerequisites

- **Python 3.8 or higher** installed on your system
- **Web browser** (Chrome, Firefox, Safari, Edge)
- **Microphone** (optional, for voice input features)

## 🎯 Quick Start (3 Methods)

### Method 1: One-Click Start (Windows)
```bash
# Double-click this file in Windows Explorer:
start_project.bat
```

### Method 2: Python Script
```bash
# Run the startup script:
python start_project.py
```

### Method 3: Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize database
python init_db.py

# 3. Start backend server
cd backend
python app.py

# 4. Open frontend in browser
# Navigate to: frontend/login.html
```

## 🌐 Accessing the Application

### Frontend URLs:
- **Login Page**: `frontend/login.html`
- **Main App**: `frontend/index.html`

### Backend Server:
- **API Endpoint**: `http://localhost:5000`
- **Status Check**: `http://localhost:5000/check-auth`

## 👤 First Time Setup

1. **Open the login page** in your browser
2. **Click "Register"** to create a new account
3. **Fill in your details**:
   - Username (unique)
   - Email address
   - Password (minimum 6 characters)
4. **Click "Create Account"**
5. **Login** with your credentials
6. **Start using the app!**

## 🔧 Features Available

### ✅ User Authentication
- Secure registration and login
- Session management
- User profiles with statistics

### ✅ Voice Recognition
- Click the 🎤 microphone button
- Speak your news text
- Automatic speech-to-text conversion

### ✅ AI-Powered Detection
- Machine learning model with 99.25% accuracy
- Real-time news classification
- Confidence scores and explanations

### ✅ Detailed Analysis
- Model reasoning explanations
- Text characteristics analysis
- Key contributing factors
- Confidence breakdown

## 🛠️ Troubleshooting

### Backend Server Issues
```bash
# Check if server is running
netstat -an | findstr :5000

# Restart server
cd backend
python app.py
```

### Database Issues
```bash
# Reinitialize database
python init_db.py
```

### Frontend Issues
- Clear browser cache
- Try opening in incognito/private mode
- Check browser console for errors

### Voice Recognition Issues
- Allow microphone access when prompted
- Try different browsers (Chrome works best)
- Check microphone permissions in browser settings

## 📁 Project Structure

```
Fake-News-Detection/
├── backend/
│   ├── app.py              # Flask backend server
│   ├── model.pkl           # Trained ML model
│   ├── vectorizer.pkl      # Text vectorizer
│   └── users.db            # User database
├── frontend/
│   ├── index.html          # Main application
│   ├── login.html          # Login/registration page
│   ├── style.css           # Main styles
│   ├── login.css           # Login page styles
│   ├── script.js           # Main JavaScript
│   └── login.js            # Login JavaScript
├── data/
│   ├── Fake.csv            # Training data
│   └── True.csv            # Training data
├── requirements.txt        # Python dependencies
├── init_db.py             # Database initialization
├── start_project.py       # Startup script
├── start_project.bat      # Windows batch file
└── README.md              # Detailed documentation
```

## 🔒 Security Notes

- Passwords are securely hashed using Werkzeug
- Sessions are managed with Flask-Login
- CORS is configured for local development
- Database uses SQLite for simplicity

## 🌟 Browser Compatibility

- **Chrome/Edge**: Full support (recommended)
- **Firefox**: Good support
- **Safari**: Limited voice recognition
- **Mobile**: Responsive design works on mobile

## 📞 Support

If you encounter issues:

1. Check the browser console for errors
2. Verify all dependencies are installed
3. Ensure the backend server is running
4. Check the README.md for detailed information

## 🎉 Enjoy Your Fake News Detection System!

The application is now ready to use. Start by registering an account and testing the fake news detection features!
