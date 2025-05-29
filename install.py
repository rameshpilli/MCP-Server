#!/usr/bin/env python3
"""
MCP Server Installation Script

This script provides a unified way to install the MCP server and its dependencies
using various package managers (pip, uvx, npm) with proper error handling and
user-friendly output.

Usage:
    python install.py [options]

Options:
    --method METHOD     Installation method: pip, uvx, npm (default: auto-detect)
    --dev               Install in development mode
    --venv PATH         Create or use a virtual environment at PATH
    --no-deps           Skip installing dependencies
    --upgrade           Upgrade existing installation
    --global            Install globally (system-wide)
    --verbose           Show verbose output
    --help              Show this help message and exit

Examples:
    python install.py                   # Auto-detect best method
    python install.py --method pip      # Use pip
    python install.py --method uvx      # Use uvx
    python install.py --method npm      # Use npm
    python install.py --dev --venv .venv  # Dev install in virtual environment
"""

import os
import sys
import argparse
import subprocess
import platform
import shutil
import venv
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

# Constants
PYTHON_PACKAGE_NAME = "crm-mcp-server"
NPM_PACKAGE_NAME = "mcp-server"
MIN_PYTHON_VERSION = (3, 9)
MIN_NODE_VERSION = (14, 0)
MIN_NPM_VERSION = (6, 0)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("mcp_installer")

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def colorize(text: str, color: str) -> str:
    """Add color to text if terminal supports it"""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.ENDC}"
    return text

def print_header(text: str) -> None:
    """Print a formatted header"""
    logger.info("\n" + colorize(f"=== {text} ===", Colors.HEADER + Colors.BOLD))

def print_step(text: str) -> None:
    """Print a step in the installation process"""
    logger.info(colorize(f"➤ {text}", Colors.BLUE))

def print_success(text: str) -> None:
    """Print a success message"""
    logger.info(colorize(f"✓ {text}", Colors.GREEN))

def print_warning(text: str) -> None:
    """Print a warning message"""
    logger.info(colorize(f"⚠ {text}", Colors.YELLOW))

def print_error(text: str) -> None:
    """Print an error message"""
    logger.error(colorize(f"✗ {text}", Colors.RED))

def run_command(
    cmd: List[str], 
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = False,
    verbose: bool = False
) -> Tuple[int, str, str]:
    """
    Run a command and return exit code, stdout, and stderr
    
    Args:
        cmd: Command to run as a list of strings
        cwd: Working directory
        env: Environment variables
        capture_output: Whether to capture output
        verbose: Whether to print command and output
        
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    if verbose:
        logger.info(f"Running: {' '.join(cmd)}")
    
    # Prepare environment
    cmd_env = os.environ.copy()
    if env:
        cmd_env.update(env)
    
    # Run command
    try:
        if capture_output:
            result = subprocess.run(
                cmd, 
                cwd=cwd, 
                env=cmd_env,
                text=True,
                capture_output=True
            )
            stdout, stderr = result.stdout, result.stderr
        else:
            # Stream output to console
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=cmd_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            stdout_lines = []
            stderr_lines = []
            
            # Process output in real-time
            for line in process.stdout:
                line = line.rstrip()
                stdout_lines.append(line)
                if verbose:
                    print(line)
            
            # Get any remaining stderr
            stderr = process.communicate()[1]
            stderr_lines = stderr.splitlines()
            
            stdout = "\n".join(stdout_lines)
            stderr = "\n".join(stderr_lines)
            
            process.wait()
            returncode = process.returncode
            
            return returncode, stdout, stderr
            
        return result.returncode, stdout, stderr
    except Exception as e:
        return 1, "", str(e)

def check_python_version() -> bool:
    """Check if Python version meets requirements"""
    version_info = sys.version_info
    if version_info < MIN_PYTHON_VERSION:
        print_error(f"Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ is required")
        print_warning(f"Current version: {version_info.major}.{version_info.minor}.{version_info.micro}")
        return False
    return True

def check_node_version() -> Tuple[bool, Optional[Tuple[int, int, int]]]:
    """Check if Node.js version meets requirements"""
    try:
        returncode, stdout, stderr = run_command(["node", "--version"], capture_output=True)
        if returncode != 0:
            print_error("Node.js is not installed or not in PATH")
            return False, None
        
        # Parse version (format: v14.17.0)
        version_str = stdout.strip().lstrip('v')
        version_parts = version_str.split('.')
        version = tuple(int(part) for part in version_parts[:3])
        
        if version < MIN_NODE_VERSION:
            print_error(f"Node.js {MIN_NODE_VERSION[0]}.{MIN_NODE_VERSION[1]}+ is required")
            print_warning(f"Current version: {version[0]}.{version[1]}.{version[2]}")
            return False, version
        
        return True, version
    except Exception as e:
        print_error(f"Error checking Node.js version: {e}")
        return False, None

def check_npm_version() -> Tuple[bool, Optional[Tuple[int, int, int]]]:
    """Check if npm version meets requirements"""
    try:
        returncode, stdout, stderr = run_command(["npm", "--version"], capture_output=True)
        if returncode != 0:
            print_error("npm is not installed or not in PATH")
            return False, None
        
        # Parse version (format: 6.14.13)
        version_str = stdout.strip()
        version_parts = version_str.split('.')
        version = tuple(int(part) for part in version_parts[:3])
        
        if version < MIN_NPM_VERSION:
            print_error(f"npm {MIN_NPM_VERSION[0]}.{MIN_NPM_VERSION[1]}+ is required")
            print_warning(f"Current version: {version[0]}.{version[1]}.{version[2]}")
            return False, version
        
        return True, version
    except Exception as e:
        print_error(f"Error checking npm version: {e}")
        return False, None

def create_virtual_environment(venv_path: Path, verbose: bool = False) -> bool:
    """Create a Python virtual environment"""
    try:
        print_step(f"Creating virtual environment at {venv_path}")
        venv.create(venv_path, with_pip=True, upgrade_deps=True)
        print_success(f"Virtual environment created at {venv_path}")
        return True
    except Exception as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False

def get_venv_python(venv_path: Path) -> str:
    """Get the path to the Python executable in a virtual environment"""
    if platform.system() == "Windows":
        return str(venv_path / "Scripts" / "python.exe")
    return str(venv_path / "bin" / "python")

def get_venv_pip(venv_path: Path) -> str:
    """Get the path to the pip executable in a virtual environment"""
    if platform.system() == "Windows":
        return str(venv_path / "Scripts" / "pip.exe")
    return str(venv_path / "bin" / "pip")

def detect_project_root() -> Optional[Path]:
    """Detect the project root directory"""
    # Start with current directory
    current_dir = Path.cwd()
    
    # Check if we're in the project root
    if (current_dir / "pyproject.toml").exists() and (current_dir / "package.json").exists():
        return current_dir
    
    # Check parent directories
    for parent in current_dir.parents:
        if (parent / "pyproject.toml").exists() and (parent / "package.json").exists():
            return parent
    
    # Check if this script is in the project
    script_dir = Path(__file__).parent.absolute()
    if (script_dir / "pyproject.toml").exists() and (script_dir / "package.json").exists():
        return script_dir
    
    # Not found
    return None

def install_with_pip(
    dev_mode: bool = False,
    venv_path: Optional[Path] = None,
    skip_deps: bool = False,
    upgrade: bool = False,
    global_install: bool = False,
    verbose: bool = False
) -> bool:
    """Install using pip"""
    print_header("Installing with pip")
    
    # Determine pip command
    if venv_path:
        pip_cmd = get_venv_pip(venv_path)
    else:
        pip_cmd = "pip"
    
    # Determine project root
    project_root = detect_project_root()
    if not project_root and not global_install:
        print_error("Could not detect project root directory")
        print_warning("Use --global to install from PyPI instead of local directory")
        return False
    
    # Build command
    cmd = [pip_cmd, "install"]
    
    if upgrade:
        cmd.append("--upgrade")
    
    if skip_deps:
        cmd.append("--no-deps")
    
    if dev_mode and project_root:
        cmd.extend(["-e", str(project_root)])
    elif global_install:
        cmd.append(PYTHON_PACKAGE_NAME)
    elif project_root:
        cmd.append(str(project_root))
    else:
        print_error("No installation source specified")
        return False
    
    # Run installation
    print_step("Running pip installation")
    returncode, stdout, stderr = run_command(cmd, verbose=verbose)
    
    if returncode != 0:
        print_error(f"pip installation failed")
        if stderr:
            print_error(stderr)
        return False
    
    print_success("pip installation completed successfully")
    
    # Install extra dependencies if needed
    if not skip_deps and project_root and (project_root / "requirements-extra.txt").exists():
        print_step("Installing extra dependencies")
        extra_cmd = [pip_cmd, "install", "-r", str(project_root / "requirements-extra.txt")]
        if upgrade:
            extra_cmd.append("--upgrade")
        
        returncode, stdout, stderr = run_command(extra_cmd, verbose=verbose)
        
        if returncode != 0:
            print_warning("Extra dependencies installation failed (non-critical)")
            if verbose and stderr:
                print_warning(stderr)
        else:
            print_success("Extra dependencies installed successfully")
    
    return True

def install_with_uvx(
    dev_mode: bool = False,
    skip_deps: bool = False,
    upgrade: bool = False,
    global_install: bool = False,
    verbose: bool = False
) -> bool:
    """Install using uvx"""
    print_header("Installing with uvx")
    
    # Check if uvx is installed
    returncode, stdout, stderr = run_command(["uvx", "--version"], capture_output=True)
    if returncode != 0:
        # Try to install uvx first
        print_step("uvx not found, attempting to install it")
        returncode, stdout, stderr = run_command(["pip", "install", "uvx"], verbose=verbose)
        
        if returncode != 0:
            print_error("Failed to install uvx")
            if stderr:
                print_error(stderr)
            return False
        
        print_success("uvx installed successfully")
    
    # Build command
    cmd = ["uvx", "install"]
    
    if dev_mode:
        cmd.append("--dev")
    
    if skip_deps:
        cmd.append("--no-deps")
    
    if upgrade:
        cmd.append("--upgrade")
    
    if global_install:
        cmd.append("--global")
    
    cmd.append(PYTHON_PACKAGE_NAME)
    
    # Run installation
    print_step("Running uvx installation")
    returncode, stdout, stderr = run_command(cmd, verbose=verbose)
    
    if returncode != 0:
        print_error("uvx installation failed")
        if stderr:
            print_error(stderr)
        return False
    
    print_success("uvx installation completed successfully")
    return True

def install_with_npm(
    dev_mode: bool = False,
    skip_deps: bool = False,
    upgrade: bool = False,
    global_install: bool = False,
    verbose: bool = False
) -> bool:
    """Install using npm"""
    print_header("Installing with npm")
    
    # Check Node.js and npm versions
    node_ok, node_version = check_node_version()
    npm_ok, npm_version = check_npm_version()
    
    if not node_ok or not npm_ok:
        print_error("Node.js or npm requirements not met")
        return False
    
    # Determine project root
    project_root = detect_project_root()
    
    # Build command
    cmd = ["npm", "install"]
    
    if upgrade:
        cmd = ["npm", "update"]
    
    if global_install:
        cmd.append("-g")
        cmd.append(NPM_PACKAGE_NAME)
    elif project_root:
        # Local installation
        if dev_mode:
            cmd.append("--save-dev")
        
        # If we're in the project root, don't specify package
        if Path.cwd() != project_root:
            os.chdir(project_root)
    else:
        # No project root found, install from npm registry
        if not global_install:
            print_warning("No project root found, installing from npm registry")
        cmd.append(NPM_PACKAGE_NAME)
    
    # Run installation
    print_step("Running npm installation")
    returncode, stdout, stderr = run_command(cmd, verbose=verbose)
    
    if returncode != 0:
        print_error("npm installation failed")
        if stderr:
            print_error(stderr)
        return False
    
    print_success("npm installation completed successfully")
    
    # Install Python dependencies if needed
    if not skip_deps and project_root:
        print_step("Installing Python dependencies")
        
        # Check if package.json has Python dependencies
        package_json_path = project_root / "package.json"
        if package_json_path.exists():
            try:
                with open(package_json_path, 'r') as f:
                    package_data = json.load(f)
                
                python_deps = package_data.get("python", {}).get("dependencies", [])
                if python_deps:
                    # Create a temporary requirements file
                    temp_req_file = project_root / "temp_requirements.txt"
                    with open(temp_req_file, 'w') as f:
                        f.write("\n".join(python_deps))
                    
                    # Install Python dependencies
                    pip_cmd = ["pip", "install", "-r", str(temp_req_file)]
                    if upgrade:
                        pip_cmd.append("--upgrade")
                    
                    returncode, stdout, stderr = run_command(pip_cmd, verbose=verbose)
                    
                    # Clean up
                    temp_req_file.unlink()
                    
                    if returncode != 0:
                        print_warning("Python dependencies installation failed (non-critical)")
                        if verbose and stderr:
                            print_warning(stderr)
                    else:
                        print_success("Python dependencies installed successfully")
            except Exception as e:
                print_warning(f"Error reading package.json: {e}")
    
    return True

def auto_detect_method() -> str:
    """Auto-detect the best installation method"""
    # Check if we're in a project directory
    project_root = detect_project_root()
    
    # Check if we're in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    
    # Check if uvx is available
    uvx_available = shutil.which("uvx") is not None
    
    # Check if npm is available
    npm_available = shutil.which("npm") is not None
    
    # Decision logic
    if project_root:
        # We're in a project directory
        if uvx_available:
            return "uvx"
        elif in_venv:
            return "pip"
        elif npm_available:
            return "npm"
        else:
            return "pip"
    else:
        # Not in a project directory
        if uvx_available:
            return "uvx"
        elif npm_available:
            return "npm"
        else:
            return "pip"

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="MCP Server Installation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1]
    )
    
    parser.add_argument(
        "--method",
        choices=["pip", "uvx", "npm", "auto"],
        default="auto",
        help="Installation method (default: auto-detect)"
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Install in development mode"
    )
    
    parser.add_argument(
        "--venv",
        metavar="PATH",
        help="Create or use a virtual environment at PATH"
    )
    
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Skip installing dependencies"
    )
    
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Upgrade existing installation"
    )
    
    parser.add_argument(
        "--global",
        dest="global_install",
        action="store_true",
        help="Install globally (system-wide)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_header("MCP Server Installation")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Handle virtual environment
    venv_path = None
    if args.venv:
        venv_path = Path(args.venv)
        if not venv_path.exists():
            if not create_virtual_environment(venv_path, verbose=args.verbose):
                sys.exit(1)
        else:
            print_step(f"Using existing virtual environment at {venv_path}")
    
    # Auto-detect method if needed
    method = args.method
    if method == "auto":
        method = auto_detect_method()
        print_step(f"Auto-detected installation method: {method}")
    
    # Install based on method
    success = False
    if method == "pip":
        success = install_with_pip(
            dev_mode=args.dev,
            venv_path=venv_path,
            skip_deps=args.no_deps,
            upgrade=args.upgrade,
            global_install=args.global_install,
            verbose=args.verbose
        )
    elif method == "uvx":
        success = install_with_uvx(
            dev_mode=args.dev,
            skip_deps=args.no_deps,
            upgrade=args.upgrade,
            global_install=args.global_install,
            verbose=args.verbose
        )
    elif method == "npm":
        success = install_with_npm(
            dev_mode=args.dev,
            skip_deps=args.no_deps,
            upgrade=args.upgrade,
            global_install=args.global_install,
            verbose=args.verbose
        )
    
    if success:
        print_header("Installation Successful")
        
        # Print usage instructions
        print_step("Usage Instructions:")
        
        if args.global_install:
            print(colorize("# Run the MCP server (API only)", Colors.BLUE))
            print(f"uvx crm-mcp-server --host 0.0.0.0 --port 8080")
            print()
            print(colorize("# Run the MCP server with UI", Colors.BLUE))
            print(f"uvx crm-mcp-server --host 0.0.0.0 --port 8080 --ui")
            print()
            print(colorize("# Run in stdio mode (for embedding)", Colors.BLUE))
            print(f"uvx crm-mcp-server --mode stdio")
            print()
            print(colorize("# Alternative npm commands", Colors.BLUE))
            print(f"npx mcp-server --host 0.0.0.0 --port 8080")
            print(f"npx mcp-server --mode stdio")
        else:
            if venv_path:
                activate_cmd = "source .venv/bin/activate" if platform.system() != "Windows" else ".venv\\Scripts\\activate"
                print(colorize(f"# Activate the virtual environment", Colors.BLUE))
                print(activate_cmd)
                print()
                
            print(colorize("# Run the MCP server locally", Colors.BLUE))
            print(f"uvx run")
            print()
            print(colorize("# Run mock financial endpoints", Colors.BLUE))
            print(f"uvx mock")
        
        sys.exit(0)
    else:
        print_header("Installation Failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
