#!/bin/bash

# Text colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    echo -e "${2}${1}${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

print_message "=================================================" "$YELLOW"
print_message "First Time Setup - No-Code ML Platform" "$YELLOW"
print_message "=================================================" "$YELLOW"
echo ""

# Check if Python is installed
if ! command_exists python3; then
    print_message "Python 3 is not installed. Please install Python 3 to use this application." "$RED"
    print_message "Visit https://www.python.org/downloads/ to download Python 3." "$YELLOW"
    exit 1
fi

# Check if pip is installed
if ! command_exists pip3; then
    print_message "pip3 is not installed. Please install pip3 to use this application." "$RED"
    print_message "Visit https://pip.pypa.io/en/stable/installation/ for instructions." "$YELLOW"
    exit 1
fi

# Remove existing virtual environment if it exists
if [ -d "venv" ]; then
    print_message "Removing existing virtual environment..." "$YELLOW"
    rm -rf venv
fi

# Create new virtual environment
print_message "Creating virtual environment..." "$YELLOW"
python3 -m venv venv
if [ $? -ne 0 ]; then
    print_message "Failed to create virtual environment. Please check your Python installation." "$RED"
    exit 1
fi

# Activate virtual environment
print_message "Activating virtual environment..." "$YELLOW"
source venv/bin/activate

# Upgrade pip
print_message "Upgrading pip..." "$YELLOW"
pip install --upgrade pip

# Install all requirements
print_message "Installing required packages..." "$YELLOW"
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    print_message "Failed to install requirements. Please check your internet connection and try again." "$RED"
    exit 1
fi

# Verify installations
print_message "Verifying installations..." "$YELLOW"
python3 -c "import streamlit; import pandas; import numpy; import sklearn; import plotly; import matplotlib; import seaborn" 2>/dev/null
if [ $? -ne 0 ]; then
    print_message "Some packages failed to install correctly. Please try running the script again." "$RED"
    exit 1
fi

# Create start script
print_message "Creating start script..." "$YELLOW"
cat > start_app.sh << 'EOF'
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
EOF

# Make start script executable
chmod +x start_app.sh

print_message "=================================================" "$GREEN"
print_message "First time setup completed successfully!" "$GREEN"
print_message "=================================================" "$GREEN"
print_message "To start the application, run: ./start_app.sh" "$GREEN"
print_message "=================================================" "$GREEN"

# Deactivate virtual environment
deactivate 