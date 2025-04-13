#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting MCP setup...${NC}"

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$python_version < 3.8" | bc -l) )); then
    echo -e "${RED}Error: Python 3.8 or higher is required${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install package with test dependencies
echo -e "${YELLOW}Installing package...${NC}"
pip install -e ".[test]"

# Run directory setup script
echo -e "${YELLOW}Setting up directories...${NC}"
python scripts/setup_directories.py

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        # Generate a secure random key
        SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
        # Replace the example secret key with the generated one
        sed -i.bak "s/your-secret-key-here/${SECRET_KEY}/" .env
        rm -f .env.bak
    else
        echo -e "${RED}Error: .env.example file not found${NC}"
        exit 1
    fi
fi

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
pytest tests/ -v

echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${YELLOW}You can now start the application with:${NC}"
echo -e "    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000" 