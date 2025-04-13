import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.absolute())
sys.path.insert(0, project_root)

# Import pytest fixtures that should be available to all tests
pytest_plugins = [
    "tests.conftest",
] 