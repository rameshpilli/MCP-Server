from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os

@dataclass
class FileInfo:
    name: str
    path: Path
    size: int
    created_at: datetime
    modified_at: datetime
    is_directory: bool

    @classmethod
    def from_path(cls, path: Path) -> 'FileInfo':
        stats = os.stat(path)
        return cls(
            name=path.name,
            path=path,
            size=stats.st_size,
            created_at=datetime.fromtimestamp(stats.st_ctime),
            modified_at=datetime.fromtimestamp(stats.st_mtime),
            is_directory=path.is_dir()
        ) 