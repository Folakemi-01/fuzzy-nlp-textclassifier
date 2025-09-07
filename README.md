# Improving Multi-class Text Classification Using Fuzzy Logic and NLP

**Author:** Afolakemi Mariam Kazeem
**Student ID** 24832326
**Course:** MSc Data Science
**Project Supervisor:** Dr Naomi Adel 


# Improving Multi-class Text Classification Using Fuzzy Logic

This repository contains the source code, experimental data, and trained models for the Master's dissertation project titled "Improving Multi-class Text Classification Accuracy Using Fuzzy Logic and NLP Techniques".

## Project Overview

This project investigates a novel approach to improving text classification on ambiguous documents by integrating a knowledge-based fuzzy logic system with a state-of-the-art BERT model. The research includes a quantitative evaluation of the hybrid model's performance, statistical significance testing, and a qualitative, human-centred study on its perceived usability.

## Setup and Installation

The project was developed using Python 3.9+.

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

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### Main Experiment

To retrain the baseline model and run the main evaluation (as reported in Section 5.2 of the dissertation), execute the following command from the root directory:

```bash
python main.py --train --evaluate