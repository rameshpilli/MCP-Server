from setuptools import setup, find_packages
import os
from pathlib import Path

def run_setup_script():
    """Run the directory setup script."""
    setup_script = Path(__file__).parent / "scripts" / "setup_directories.py"
    if setup_script.exists():
        exec(setup_script.read_text())

setup(
    name="mcp",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "fastapi",
        "uvicorn",
        "sqlalchemy",
        "alembic",
        "aiosqlite",
        "python-dotenv",
        "pytest>=8.3.5",
        "httpx>=0.27.0",
        "python-jose>=3.4.0",
        "passlib>=1.7.4",
        "python-multipart>=0.0.20",
        "structlog>=24.1.0",
        "python-json-logger>=2.0.7",
        "jinja2>=3.1.0",
        "aiofiles>=23.2.0",
    ],
    extras_require={
        "test": [
            "pytest>=8.3.5",
            "pytest-asyncio>=0.23.5",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mcp=app.main:start",
        ],
    },
)

# Run setup script after installation
run_setup_script() 