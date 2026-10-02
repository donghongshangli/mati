"""
Resource Path Resolution Utility

Provides PyInstaller-compatible path resolution for resources.
Handles both development and bundled executable environments.
"""

import os
import sys
from pathlib import Path


def get_base_path() -> Path:
    """
    Get the base path for resources, handling both development and PyInstaller environments.
    
    Returns:
        Path: Base path to the application resources
    """
    if getattr(sys, 'frozen', False):
        base_path = Path(sys._MEIPASS)
    else:
        this_file = Path(__file__).resolve()
        base_path = this_file.parent.parent.parent
    
    return base_path


def resolve_content_dir(relative_path: str) -> Path:
    """
    Resolve the content directory path.
    
    Args:
        relative_path (str): Relative path from base (e.g., "mati_data/content")
        
    Returns:
        Path: Resolved absolute path to content directory
    """
    base_path = get_base_path()
    resolved = base_path / relative_path
    
    if not resolved.exists() and not getattr(sys, 'frozen', False):
        project_root = base_path
        resolved = project_root / relative_path
        
        if not resolved.exists():
            fallback = project_root / "scripts" / "data_collection" / "data" / "content"
            if fallback.exists():
                return fallback
    
    return resolved


def resolve_model_dir(relative_path: str) -> Path:
    """
    Resolve the model directory path.
    
    Args:
        relative_path (str): Relative path from base (e.g., "mati_data/models/qwen2_5")
        
    Returns:
        Path: Resolved absolute path to model directory
    """
    base_path = get_base_path()
    resolved = base_path / relative_path
    
    if not resolved.exists() and not getattr(sys, 'frozen', False):
        project_root = base_path
        resolved = project_root / relative_path
    
    return resolved


def resolve_chroma_db_dir(relative_path: str = "mati_data/chroma_db") -> Path:
    """
    Resolve the ChromaDB directory path.
    
    Args:
        relative_path (str): Relative path from base (default: "mati_data/chroma_db")
        
    Returns:
        Path: Resolved absolute path to ChromaDB directory
    """
    base_path = get_base_path()
    resolved = base_path / relative_path
    
    if not resolved.exists() and not getattr(sys, 'frozen', False):
        project_root = base_path
        resolved = project_root / relative_path
    
    return resolved

