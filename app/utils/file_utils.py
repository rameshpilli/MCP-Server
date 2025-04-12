import os
from pathlib import Path
from typing import List, Optional
from app.core.config import ServerConfig

def is_allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ServerConfig.ALLOWED_EXTENSIONS

def get_safe_path(base_dir: Path, path: str) -> Path:
    """Get a safe absolute path that doesn't escape the base directory."""
    try:
        safe_path = (base_dir / path).resolve()
        if not str(safe_path).startswith(str(base_dir)):
            raise ValueError("Path attempts to escape base directory")
        return safe_path
    except (ValueError, RuntimeError) as e:
        raise ValueError(f"Invalid path: {e}")

def create_directory(path: Path) -> None:
    """Create a directory if it doesn't exist."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create directory: {e}")

def get_file_size(path: Path) -> int:
    """Get file size in bytes."""
    try:
        return path.stat().st_size
    except Exception as e:
        raise RuntimeError(f"Failed to get file size: {e}")

def delete_file(path: Path) -> None:
    """Delete a file or empty directory."""
    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
        else:
            raise ValueError("Can only delete files or empty directories")
    except Exception as e:
        raise RuntimeError(f"Failed to delete: {e}")

def list_directory(path: Path, recursive: bool = False) -> List[Path]:
    """List contents of a directory."""
    try:
        if recursive:
            return list(path.rglob('*'))
        return list(path.iterdir())
    except Exception as e:
        raise RuntimeError(f"Failed to list directory: {e}")

def get_mime_type(path: Path) -> str:
    """Get MIME type of a file."""
    import mimetypes
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or 'application/octet-stream' 