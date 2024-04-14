import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", PACKAGE_DIR / "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PACKAGE_DIR / "output"))
