@echo off
setlocal enabledelayedexpansion

echo Starting MCP setup...

REM Check Python version
python -c "import sys; assert sys.version_info >= (3,8), 'Python 3.8+ is required'" 2>nul
if errorlevel 1 (
    echo Error: Python 3.8 or higher is required
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install package with test dependencies
echo Installing package...
pip install -e .[test]

REM Run directory setup script
echo Setting up directories...
python scripts\setup_directories.py

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file from template...
    if exist .env.example (
        copy .env.example .env
        REM Generate a secure random key
        for /f "tokens=*" %%i in ('python -c "import secrets; print(secrets.token_hex(32))"') do set SECRET_KEY=%%i
        REM Replace the example secret key with the generated one
        powershell -Command "(Get-Content .env) -replace 'your-secret-key-here', '%SECRET_KEY%' | Set-Content .env"
    ) else (
        echo Error: .env.example file not found
        exit /b 1
    )
)

REM Run tests
echo Running tests...
pytest tests\ -v

echo Setup completed successfully!
echo You can now start the application with:
echo     uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

endlocal 