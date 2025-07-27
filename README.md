# Improving Multi-class Text Classification Using Fuzzy Logic and NLP

**Author:** Afolakemi Mariam Kazeem
**Student ID** 24832326
**Course:** MSc Data Science
**Project Supervisor:** Dr Naomi Adel 


---

## 1. Project Overview

This project investigates a novel hybrid architecture for multi-class text classification, aiming to improve performance in contexts of high linguistic ambiguity. The core hypothesis is that integrating a fuzzy inference system with a Transformer-based model can yield more accurate and interpretable results than a standard deep learning approach alone.

The research implements and evaluates a `FuzzyBertClassifier`, which applies a targeted, knowledge-based fuzzy rule set to the softmax probability outputs of a fine-tuned BERT model. This hybrid model is rigorously benchmarked against the standard `bert-base-uncased` classifier on the 20 Newsgroups dataset, a classic benchmark known for its thematic overlap between categories.

The findings demonstrate that a simple, manually engineered fuzzy rule is insufficient to improve upon the baseline and can be detrimental to performance, highlighting the complexity of intervening in the decision-making process of deep neural networks.

## 2. Project Structure

The repository is organized into a modular structure to ensure clarity, maintainability, and full reproducibility of the experimental results.

-   **/src**: Contains all Python source code.
    -   `models/`: Contains the class definitions for `BaselineBertClassifier` and `FuzzyClassifier`.
    -   `data_loader.py`: Handles loading and splitting of the dataset.
    -   `preprocessing.py`: Contains the text cleaning and preprocessing function.
    -   `runner.py`: The main `ExperimentRunner` class that orchestrates the entire training and prediction pipeline.
    -   `evaluation.py`: The `Evaluation` class, which takes the runner's output to generate all metrics and visualisations.
-   **/results**: The designated output directory for all generated files, including performance metrics (`.csv`) and confusion matrices (`.png`).
-   **/models**: The designated output directory for the saved, trained baseline model (`.pth` file).
-   `config.py`: A centralized configuration file for all project parameters (e.g., file paths, model hyperparameters).
-   `main.py`: The single entry point to execute the full experiment and evaluation pipeline.
-   `requirements.txt`: A list of all Python dependencies required to run the project.

## 3. Setup and Execution

### Prerequisites
- Python 3.10+
- Git

### Installation
To set up the environment and run this project, follow these steps from your terminal:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Folakemi-01/fuzzy-nlp-textclassifier.git](https://github.com/Folakemi-01/fuzzy-nlp-textclassifier.git)
    cd fuzzy-nlp-textclassifier
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install all required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Experiment
The entire pipeline—from data loading to final evaluation—can be executed with a single command:

```bash
python main.py