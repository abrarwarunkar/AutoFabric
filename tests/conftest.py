"""
AutoFabric pytest configuration.
Sets sys.path so that `backend.` imports work from tests/ directory.
"""
import sys
from pathlib import Path

# Add autofabric/ root to sys.path so `from backend.x import y` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
