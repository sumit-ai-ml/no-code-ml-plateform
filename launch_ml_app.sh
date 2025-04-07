#!/bin/bash

# Text colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print colored messages
print_message() {
    echo -e "${2}${1}${NC}"
}

# Check if Python is installed
if ! command_exists python3; then
    print_message "Python 3 is not installed. Please install Python 3 to use this application." "$RED"
    exit 1
fi

# Check if pip is installed
if ! command_exists pip3; then
    print_message "pip3 is not installed. Please install pip3 to use this application." "$RED"
    exit 1
fi

# Check if virtual environment exists, if not create it
if [ ! -d "venv" ]; then
    print_message "Creating virtual environment..." "$YELLOW"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        print_message "Failed to create virtual environment. Please check your Python installation." "$RED"
        exit 1
    fi
fi

# Activate virtual environment
print_message "Activating virtual environment..." "$YELLOW"
source venv/bin/activate

# Install requirements if needed
if [ ! -f "venv/lib/python3.*/site-packages/streamlit" ]; then
    print_message "Installing required packages..." "$YELLOW"
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        print_message "Failed to install requirements. Please check your internet connection and try again." "$RED"
        exit 1
    fi
fi

# Check if Streamlit is installed
if ! command_exists streamlit; then
    print_message "Installing Streamlit..." "$YELLOW"
    pip install streamlit
    if [ $? -ne 0 ]; then
        print_message "Failed to install Streamlit. Please check your internet connection and try again." "$RED"
        exit 1
    fi
fi

# Launch the application
print_message "Starting No-Code ML Platform..." "$GREEN"
print_message "The application will open in your default web browser." "$GREEN"
print_message "Press Ctrl+C to stop the application." "$YELLOW"

# Run the Streamlit app
streamlit run app.py

# Deactivate virtual environment when done
deactivate 