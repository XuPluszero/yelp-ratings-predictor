@echo off
REM Quick Start Script for Yelp Restaurant Rating Predictor (Windows)

echo ======================================
echo Yelp Rating Predictor - Quick Setup
echo ======================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo X Python is not installed. Please install Python 3.7+ first.
    pause
    exit /b 1
)

echo + Python found
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

echo + Virtual environment created and activated

REM Install requirements
echo.
echo Installing dependencies...
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q

echo + Dependencies installed

REM Check for data files
echo.
echo Checking for data files...
if not exist "yelp242a_train.csv" (
    echo ! Data files not found!
    echo.
    echo Please download the following files and place them in this directory:
    echo   - yelp242a_train.csv
    echo   - yelp242a_test.csv
    echo.
    echo See DATA_README.md for more information on obtaining the data.
    echo.
) else (
    echo + Data files found
)

REM Create outputs directory
if not exist "outputs" mkdir outputs
echo + Output directory ready

echo.
echo ======================================
echo Setup Complete!
echo ======================================
echo.
echo To run the analysis:
echo   1. Make sure your data files are in place
echo   2. Run: python predict_ratings.py
echo.
echo To deactivate virtual environment later:
echo   Run: deactivate
echo.
pause
