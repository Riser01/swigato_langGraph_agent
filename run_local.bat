@echo off
REM Build and run the chatbot application locally on Windows

echo 🤖 AI Chatbot MVP - Local Setup
echo ================================

REM Check if .env file exists
if not exist .env (
    echo ❌ .env file not found!
    echo 📋 Creating .env from template...
    copy .env.example .env
    echo ✅ Please edit .env file and add your OpenAI API key
    echo    Then run this script again.
    pause
    exit /b 1
)

REM Check if OPENAI_API_KEY is set
findstr /C:"OPENAI_API_KEY=sk-" .env >nul
if errorlevel 1 (
    echo ❌ OpenAI API key not found in .env file!
    echo 📝 Please add your OpenAI API key to the .env file
    echo    Example: OPENAI_API_KEY=sk-your-api-key-here
    pause
    exit /b 1
)

echo 🔧 Setting up Python environment...

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo 🐍 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Create logs directory
if not exist logs mkdir logs

echo 🚀 Starting the chatbot application...
echo 🌐 The app will be available at: http://localhost:8501
echo 🛑 Press Ctrl+C to stop the application
echo.

REM Run the Streamlit app
streamlit run app.py
