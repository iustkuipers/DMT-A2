"""
Data Module for Assignment 2
Entry point for loading and analyzing data
"""

from pathlib import Path
from .eda import HotelEDA


class DataHandler:
    """
    Modular data handler for Expedia hotel ranking data.
    Coordinates loading, preprocessing, and EDA.
    """
    
    def __init__(self, data_dir="Data"):
        """
        Initialize with data directory path.
        
        Args:
            data_dir: path to Data folder containing Raw/
        """
        self.data_dir = Path(data_dir).resolve()
        self.train_path = self.data_dir / "Raw" / "training_set_VU_DM.csv"
        self.test_path = self.data_dir / "Raw" / "test_set_VU_DM.csv"
        self.eda = None
    
    def run_eda(self, output_dir="plots/eda"):
        """
        Run exploratory data analysis on training and test sets.
        
        Args:
            output_dir: directory to save EDA outputs (plots, CSVs)
        """
        # Initialize EDA with both train and test data
        self.eda = HotelEDA(
            train_path=str(self.train_path),
            test_path=str(self.test_path) if self.test_path.exists() else None
        )
        
        # Run full EDA pipeline
        self.eda.run_full_eda(output_dir=output_dir)
        
        return self.eda
    
    def get_eda(self):
        """Get EDA object (after run_eda has been called)."""
        if self.eda is None:
            raise ValueError("EDA not yet run. Call run_eda() first.")
        return self.eda


def main(data_dir="Data", output_dir="plots/eda"):
    """
    Main entry point for data module.
    
    Args:
        data_dir: path to data directory
        output_dir: output directory for EDA results
    """
    handler = DataHandler(data_dir=data_dir)
    handler.run_eda(output_dir=output_dir)
    return handler


if __name__ == "__main__":
    handler = main()
