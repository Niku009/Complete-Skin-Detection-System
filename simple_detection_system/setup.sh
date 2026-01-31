#!/bin/bash
# Simple Setup Script

echo "🎯 Simple Detection System Setup"
echo "================================="
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt -q

# Create folders
echo "📁 Creating folders..."
mkdir -p weights
mkdir -p results

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Add your models to weights/ folder"
echo "   2. Run: python detect.py"
echo ""
