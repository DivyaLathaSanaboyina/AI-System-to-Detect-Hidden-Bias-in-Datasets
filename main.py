# main.py
"""
Master pipeline runner.
Run this file to execute the complete bias audit system end-to-end.

Usage:
    python main.py --dataset adult
    python main.py --dataset german
    python main.py --dataset adult --skip-mitigation
"""

import argparse
import logging
from config import DATASETS

# Configure logging — professional standard
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def run_pipeline(dataset_name: str):
    """
    Executes the complete bias detection and mitigation pipeline.
    
    Args:
        dataset_name: Either 'adult' or 'german'
    """
    logger.info(f"🚀 Starting Bias Audit Pipeline for: {dataset_name.upper()}")
    
    # ── Phase 2: Load Data ──────────────────────────────────────
    # from src.data_loader import DataLoader
    # loader = DataLoader(dataset_name)
    # df = loader.load()
    
    # ── Phase 3: Preprocess ─────────────────────────────────────
    # from src.preprocessor import Preprocessor
    # preprocessor = Preprocessor(dataset_name)
    # X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    
    # ── Phase 4: Train Models ───────────────────────────────────
    # from src.model_trainer import ModelTrainer
    # trainer = ModelTrainer()
    # models = trainer.train_all(X_train, y_train)
    
    # ── Phase 5 & 6: Detect Bias + Compute Metrics ──────────────
    # ... (filled in later phases)
    
    logger.info("✅ Pipeline complete. Check reports/ for output.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Bias Detection System")
    parser.add_argument(
        "--dataset",
        choices=["adult", "german"],
        default="adult",
        help="Dataset to audit"
    )
    args = parser.parse_args()
    run_pipeline(args.dataset)