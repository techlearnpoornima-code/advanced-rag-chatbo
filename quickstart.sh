#!/bin/bash
# Quick Start Script for Advanced RAG Chatbot

set -e

echo "========================================="
echo "  Advanced RAG Chatbot - Quick Start"
echo "========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your ANTHROPIC_API_KEY"
    echo ""
    read -p "Press Enter after you've added your API key to .env..."
fi

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "✓ Dependencies installed"
echo ""

# Ask if user wants to ingest data
read -p "Do you want to ingest Stack Overflow data now? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting data ingestion..."
    echo "This will download ~1000 Stack Overflow Q&A on Python, JavaScript, SQL, Docker, FastAPI"
    echo ""
    python scripts/ingest_data.py --tags python javascript sql docker fastapi --limit 1000
fi

echo ""
echo "========================================="
echo "  Ready to Run!"
echo "========================================="
echo ""
echo "Start the chatbot with:"
echo "  python app/main.py"
echo ""
echo "Or use uvicorn directly:"
echo "  uvicorn app.main:app --reload"
echo ""
echo "API will be available at:"
echo "  http://localhost:8000"
echo ""
echo "Documentation:"
echo "  http://localhost:8000/docs"
echo ""
echo "========================================="
