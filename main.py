# main.py

from src.runner import ExperimentRunner
from src.evaluation import Evaluation
from src.data_loader import get_class_names

if __name__ == "__main__":
    # 1. Run the entire experiment pipeline
    # This will train models and save a results.csv file
    runner = ExperimentRunner()
    runner.run()
    
    # 2. Run the evaluation on the results
    # This will generate reports and confusion matrices from results.csv
    class_names = get_class_names()
    evaluator = Evaluation(class_names=class_names)
    evaluator.run_evaluation()