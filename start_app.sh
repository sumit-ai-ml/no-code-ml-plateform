#!/bin/bash

# Text colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    echo -e "${2}${1}${NC}"
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_message "Virtual environment not found. Please run first_time_setup.sh first." "$YELLOW"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Launch the application
print_message "Starting No-Code ML Platform..." "$GREEN"
print_message "The application will open in your default web browser." "$GREEN"
print_message "Press Ctrl+C to stop the application." "$YELLOW"

# Run the Streamlit app
streamlit run app.py

# Deactivate virtual environment when done
deactivate
