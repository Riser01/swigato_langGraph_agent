#!/bin/bash

# Build and run the chatbot application locally

echo "🤖 AI Chatbot MVP - Local Setup"
echo "================================"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "📋 Creating .env from template..."
    cp .env.example .env
    echo "✅ Please edit .env file and add your OpenAI API key"
    echo "   Then run this script again."
    exit 1
fi

# Check if OPENAI_API_KEY is set
if ! grep -q "OPENAI_API_KEY=sk-" .env; then
    echo "❌ OpenAI API key not found in .env file!"
    echo "📝 Please add your OpenAI API key to the .env file"
    echo "   Example: OPENAI_API_KEY=sk-your-api-key-here"
    exit 1
fi

echo "🔧 Setting up Python environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create logs directory
mkdir -p logs

echo "🚀 Starting the chatbot application..."
echo "🌐 The app will be available at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

# Run the Streamlit app
streamlit run app.py
