@echo off
echo ============================================================
echo 📰 FAKE NEWS DETECTION PROJECT
echo ============================================================
echo 🚀 Starting the complete project...
echo.

echo 📦 Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo.
echo 📦 Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo 🗄️ Initializing database...
python init_db.py
if %errorlevel% neq 0 (
    echo ❌ Failed to initialize database
    pause
    exit /b 1
)

echo.
echo 🌐 Opening frontend in browser...
start frontend\login.html

echo.
echo 🔧 Starting backend server...
echo 📝 The server will start on http://localhost:5000
echo 📝 Press Ctrl+C to stop the server
echo.
cd backend
python app.py

pause
