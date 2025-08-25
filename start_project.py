#!/usr/bin/env python3
"""
Startup script for Fake News Detection Project
This script will help you run the complete project with all necessary steps.
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def print_banner():
    print("=" * 60)
    print("📰 FAKE NEWS DETECTION PROJECT")
    print("=" * 60)
    print("🚀 Starting the complete project...")
    print()

def check_dependencies():
    print("📦 Checking dependencies...")
    try:
        import flask
        import flask_sqlalchemy
        import flask_login
        import sklearn
        import pandas
        import numpy
        print("✅ All dependencies are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def initialize_database():
    print("🗄️  Initializing database...")
    try:
        from init_db import init_database
        init_database()
        print("✅ Database initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def start_backend():
    print("🔧 Starting backend server...")
    try:
        # Change to backend directory
        backend_dir = Path("backend")
        if not backend_dir.exists():
            print("❌ Backend directory not found!")
            return False
        
        # Start the Flask server
        print("🌐 Flask server starting on http://localhost:5000")
        print("📝 Press Ctrl+C to stop the server")
        print()
        
        # Run the Flask app
        os.chdir(backend_dir)
        subprocess.run([sys.executable, "app.py"])
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return False

def open_frontend():
    print("🌐 Opening frontend...")
    try:
        # Get the absolute path to the frontend files
        current_dir = Path.cwd()
        login_path = current_dir / "frontend" / "login.html"
        main_path = current_dir / "frontend" / "index.html"
        
        # Convert to file URL
        login_url = f"file:///{login_path.absolute().as_posix()}"
        main_url = f"file:///{main_path.absolute().as_posix()}"
        
        print(f"🔗 Login page: {login_url}")
        print(f"🔗 Main app: {main_url}")
        
        # Open the login page in default browser
        webbrowser.open(login_url)
        
        print("✅ Frontend opened in browser!")
        return True
    except Exception as e:
        print(f"❌ Failed to open frontend: {e}")
        return False

def main():
    print_banner()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return
    
    # Step 2: Initialize database
    if not initialize_database():
        return
    
    # Step 3: Open frontend
    open_frontend()
    
    print("\n" + "=" * 60)
    print("🎉 PROJECT SETUP COMPLETE!")
    print("=" * 60)
    print("📋 Next steps:")
    print("1. The frontend should have opened in your browser")
    print("2. If not, manually open: frontend/login.html")
    print("3. Register a new account or login")
    print("4. Start using the fake news detection features!")
    print()
    print("🔧 To start the backend server, run:")
    print("   cd backend && python app.py")
    print()
    print("📚 For more information, see README.md")
    print("=" * 60)

if __name__ == "__main__":
    main()
