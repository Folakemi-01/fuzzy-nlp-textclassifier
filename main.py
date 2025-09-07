# main.py

import argparse
from src.runner import ExperimentRunner
from src.evaluation import Evaluation
from src.data_loader import get_class_names

def main():
    # --- 1. Set up Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the Fuzzy Text Classification experiment pipeline.")
    
    parser.add_argument(
        '--train', 
        action='store_true', 
        help="Run the full training and prediction pipeline. This will overwrite existing model files and results."
    )
    
    parser.add_argument(
        '--evaluate', 
        action='store_true', 
        help="Run the evaluation on the existing results.csv file to generate reports."
    )
    
    args = parser.parse_args()
    
    # If no arguments are given, default to running both.
    if not args.train and not args.evaluate:
        print("No specific action provided. Running BOTH training and evaluation by default.")
        args.train = True
        args.evaluate = True

    # --- 2. Execute Actions Based on Arguments ---
    if args.train:
        print("\n--- Starting Experiment Runner (Training & Prediction) ---")
        runner = ExperimentRunner()
        runner.run()
        print("--- Experiment Runner Finished ---\n")
    
    if args.evaluate:
        print("\n--- Starting Evaluation ---")
        class_names = get_class_names()
        evaluator = Evaluation(class_names=class_names)
        evaluator.run_evaluation()
        print("--- Evaluation Finished ---\n")

if __name__ == "__main__":
    main()