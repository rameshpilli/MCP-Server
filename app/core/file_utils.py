from pathlib import Path
import os
import shutil
import mimetypes
from typing import List, Optional, Union
from .config import ServerConfig

def is_allowed_file(filename: str) -> bool:
    """
    Check if the file extension is in the allowed list.
    
    Args:
        filename (str): Name of the file to check
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    if not filename or '.' not in filename:
        return False
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ServerConfig.ALLOWED_EXTENSIONS

def get_safe_path(base_dir: Union[str, Path], path: str) -> Path:
    """
    Ensure the resulting path remains within the base directory.
    
    Args:
        base_dir (Union[str, Path]): Base directory path
        path (str): Requested path
        
    Returns:
        Path: Safe absolute path
        
    Raises:
        ValueError: If path attempts to escape base directory
    """
    base_path = Path(base_dir).resolve()
    full_path = (base_path / path).resolve()
    
    if not str(full_path).startswith(str(base_path)):
        raise ValueError("Access to path outside base directory is forbidden")
    
    return full_path

def create_directory(path: Union[str, Path]) -> None:
    """
    Create a directory and its parents if they don't exist.
    
    Args:
        path (Union[str, Path]): Directory path to create
        
    Raises:
        OSError: If directory creation fails
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def get_file_size(path: Union[str, Path]) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        path (Union[str, Path]): Path to the file
        
    Returns:
        int: Size of the file in bytes
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    return Path(path).stat().st_size

def delete_file(path: Union[str, Path]) -> None:
    """
    Safely delete a file or empty directory.
    
    Args:
        path (Union[str, Path]): Path to delete
        
    Raises:
        FileNotFoundError: If path doesn't exist
        OSError: If deletion fails
    """
    path = Path(path)
    if path.is_file():
        path.unlink()
    elif path.is_dir() and not any(path.iterdir()):
        path.rmdir()
    else:
        raise OSError(f"Cannot delete non-empty directory: {path}")

def list_directory(
    path: Union[str, Path],
    recursive: bool = False,
    page: int = 1
) -> List[dict]:
    """
    List contents of a directory.
    
    Args:
        path (Union[str, Path]): Directory to list
        recursive (bool): Whether to list subdirectories recursively
        page (int): Page number for pagination
        
    Returns:
        List[dict]: List of file/directory information
        
    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    
    items = []
    start_idx = (page - 1) * ServerConfig.ITEMS_PER_PAGE
    end_idx = start_idx + ServerConfig.ITEMS_PER_PAGE
    
    if recursive:
        iterator = path.rglob('*')
    else:
        iterator = path.iterdir()
    
    for item in sorted(iterator)[start_idx:end_idx]:
        items.append({
            'name': item.name,
            'path': str(item.relative_to(path)),
            'type': 'directory' if item.is_dir() else 'file',
            'size': get_file_size(item) if item.is_file() else None,
            'mime_type': get_mime_type(item) if item.is_file() else None
        })
    
    return items

def get_mime_type(path: Union[str, Path]) -> Optional[str]:
    """
    Get the MIME type of a file.
    
    Args:
        path (Union[str, Path]): Path to the file
        
    Returns:
        Optional[str]: MIME type of the file or None if not determined
    """
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type 