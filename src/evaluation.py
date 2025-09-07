# src/evaluation.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from config import RESULTS_FILE, RESULTS_DIR, METRICS_FILE, BASELINE_CM_FILE, HYBRID_CM_FILE

class Evaluation:
    def __init__(self, class_names):
        self.results_df = pd.read_csv(RESULTS_FILE)
        self.labels = self.results_df['true_label']
        self.class_names = class_names
        print("Evaluation module initialized.")

    def generate_report(self):
        print("\n" + "="*50)
        print("          PERFORMANCE REPORT: BASELINE BERT")
        print("="*50)

        baseline_preds = self.results_df['baseline_prediction']
        present_labels = np.unique(np.concatenate((self.labels.unique(), baseline_preds.unique())))
        present_target_names = [self.class_names[i] for i in present_labels]
        
        report_baseline = classification_report(
            self.labels, baseline_preds, labels=present_labels,
            target_names=present_target_names, zero_division=0
        )
        print(report_baseline)

        print("\n" + "="*50)
        print("          PERFORMANCE REPORT: FUZZY-ENHANCED")
        print("="*50)
        
        hybrid_preds = self.results_df['hybrid_prediction']
        present_labels_hybrid = np.unique(np.concatenate((self.labels.unique(), hybrid_preds.unique())))
        present_target_names_hybrid = [self.class_names[i] for i in present_labels_hybrid]
        
        report_hybrid = classification_report(
            self.labels, hybrid_preds, labels=present_labels_hybrid,
            target_names=present_target_names_hybrid, zero_division=0
        )
        print(report_hybrid)

        # --- Save detailed metrics to a file for the dissertation ---
        report_dict_baseline = classification_report(self.labels, baseline_preds, output_dict=True, zero_division=0)
        report_dict_hybrid = classification_report(self.labels, hybrid_preds, output_dict=True, zero_division=0)
        
        df_baseline = pd.DataFrame(report_dict_baseline).transpose()
        df_hybrid = pd.DataFrame(report_dict_hybrid).transpose()
        
        combined_df = pd.concat([df_baseline, df_hybrid], keys=['Baseline', 'Hybrid'], axis=1)
        combined_df.to_csv(METRICS_FILE)
        print(f"\nDetailed metrics saved to {METRICS_FILE}")


    def plot_confusion_matrix(self, model_name, save_path):
        """Generates and saves a high-quality, readable confusion matrix."""
        preds = self.results_df[f'{model_name}_prediction']
        
        
        present_labels = np.unique(np.concatenate((self.labels.unique(), preds.unique())))
        present_class_names = [self.class_names[i] for i in present_labels]
        
        cm = confusion_matrix(self.labels, preds, labels=present_labels)        

        annotate_matrix = True if len(present_labels) <= 20 else False
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=annotate_matrix, fmt='d', cmap='Blues', 
                    xticklabels=present_class_names, yticklabels=present_class_names)
        
        plt.title(f'Confusion Matrix - {model_name.capitalize()} Model', fontsize=18)
        plt.ylabel('Actual Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Clearer confusion matrix saved to {save_path}")
        plt.close()

    def run_evaluation(self):
        self.generate_report()
        self.plot_confusion_matrix('baseline', BASELINE_CM_FILE)
        self.plot_confusion_matrix('hybrid', HYBRID_CM_FILE)
        print("\nEvaluation complete.")