"""
Main script for Assignment 2 - Expedia Hotel Ranking
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from Module.data import DataHandler


def main():
    handler = DataHandler(data_dir=str(project_root / "Data"))
    handler.run_eda(output_dir=str(project_root / "plots" / "eda"))


if __name__ == "__main__":
    main()
