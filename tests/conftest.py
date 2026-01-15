import sys
from pathlib import Path

# Add the repo root to Python's import path so tests can import layers.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))