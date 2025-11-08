#!/bin/bash
# Quick Start Script for Yelp Restaurant Rating Predictor

echo "======================================"
echo "Yelp Rating Predictor - Quick Setup"
echo "======================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "✓ Virtual environment created and activated"

# Install requirements
echo ""
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "✓ Dependencies installed"

# Check for data files
echo ""
echo "Checking for data files..."
if [ ! -f "yelp242a_train.csv" ] || [ ! -f "yelp242a_test.csv" ]; then
    echo "⚠️  Data files not found!"
    echo ""
    echo "Please download the following files and place them in this directory:"
    echo "  - yelp242a_train.csv"
    echo "  - yelp242a_test.csv"
    echo ""
    echo "See DATA_README.md for more information on obtaining the data."
    echo ""
else
    echo "✓ Data files found"
fi

# Create outputs directory
mkdir -p outputs
echo "✓ Output directory ready"

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "To run the analysis:"
echo "  1. Make sure your data files are in place"
echo "  2. Run: python predict_ratings.py"
echo ""
echo "To deactivate virtual environment later:"
echo "  Run: deactivate"
echo ""
