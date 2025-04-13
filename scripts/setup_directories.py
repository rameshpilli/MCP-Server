import os
from pathlib import Path

def setup_project_directories():
    """Create all necessary directories for the project."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.absolute()
    
    # List of directories to create
    directories = [
        "static",           # For static files
        "storage",          # For file storage
        "logs",            # For log files
        "data",            # For database and other data
        "tests/data",      # For test data
        "static/css",      # For CSS files
        "static/js",       # For JavaScript files
        "static/img",      # For images
        "templates"        # For HTML templates
    ]
    
    # Create each directory if it doesn't exist
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Create necessary placeholder files
    placeholder_files = {
        "static/css/style.css": "/* Main stylesheet */\n",
        "static/js/main.js": "// Main JavaScript file\n",
        "templates/base.html": "<!DOCTYPE html>\n<html>\n<head>\n    <title>MCP</title>\n</head>\n<body>\n    {% block content %}{% endblock %}\n</body>\n</html>"
    }
    
    for file_path, content in placeholder_files.items():
        full_path = project_root / file_path
        if not full_path.exists():
            full_path.write_text(content)
            print(f"Created file: {full_path}")

if __name__ == "__main__":
    setup_project_directories() 