# Autotag

Welcome to the Autotagging Project repository! This project explores different approaches to autotagging using machine learning techniques.

## Overview

This project aims to automatically assign tags or categories to text documents. The autotagging is performed using three different approaches:

1. **BERT Approach**: Utilizes the BERT (Bidirectional Encoder Representations from Transformers) model for sequence classification.

2. **Naive Bayes Approach**: Implements a Naive Bayes classifier with CountVectorizer for word embeddings.

3. **Torch Approach**: Employs a simple neural network implemented in PyTorch for text classification.

## Dataset

The autotagging models are trained and evaluated on the BBC dataset, which consists of text documents from different news categories.

## Directory Structure

- **bert_approach**: Contains the Python script for the BERT approach.
- **naive_bayes_approach**: Contains the Python script for the Naive Bayes approach.
- **torch_approach**: Contains the Python script for the PyTorch approach.
- **data**: Holds the dataset used for training and testing, in particular, the BBC dataset (http://mlg.ucd.ie/datasets/bbc.html)
- **results**: Stores the results of the autotagging models.

# Autotagging Approaches Evaluation

## Autotag Approach with BERT

- **Accuracy:** 97.98%
- **Precision, Recall, F1-Score for each class and averages (macro avg, weighted avg):** 
  - These metrics measure the model's precision in classifying each class. Higher values generally indicate a good ability of the model to distinguish between different classes. Accuracy is the ratio of correct predictions to the total predictions. Notably, the model achieved high precision, recall, and F1-scores, indicating a robust ability to make accurate predictions and capture instances of each class effectively.

## Autotag Approach with PyTorch

- **Accuracy:** 96.40%
- **Precision, Recall, F1-Score for each class and averages (macro avg, weighted avg):**
  - Similar to the above, these metrics provide an evaluation of the model's precision for each class and overall averages. An accuracy of 96.40% indicates that 96.40% of the model's predictions are correct.
  - While the accuracy is slightly lower compared to BERT, the model still exhibits strong classification capabilities.

## Autotag Approach with Naive Bayes

- **Accuracy:** 97.30%
- **Precision, Recall, F1-Score for each class and averages (macro avg, weighted avg):**
  - Likewise, these metrics offer an assessment of the classification model's performance.
  - The Naive Bayes approach appears to be competitive, aligning with the performance of more complex models like BERT and PyTorch.

## Overall 
- Overall, a high accuracy and high values for precision, recall, and f1-score indicate that the models are performing well in this specific classification task.

- All three approaches showcase promising results, with each model demonstrating strengths in accurate classification.
- The choice between the models may depend on factors such as computational efficiency, resource requirements, and specific use-case considerations.

## Instructions

### BERT Approach

1. Navigate to the `bert_approach` directory.
2. Run the script `bert_autotagging.py`.
3. Check the results in the `results/bert_results.txt` file.

### Naive Bayes Approach

1. Navigate to the `naive_bayes_approach` directory.
2. Run the script `naive_bayes_autotagging.py`.
3. Check the results in the `results/naive_bayes_results.txt` file.

### Torch Approach

1. Navigate to the `torch_approach` directory.
2. Run the script `torch_autotagging.py`.
3. Check the results in the `results/pytorch_results.txt` file.

## Results

- Detailed results for each approach can be found in their respective result files.


## Dependencies

- Python 3.x
- Required Python packages specified in `requirements.txt`.

## Usage

1. Clone the repository.
   ```bash
   git clone https://github.com/yourusername/autotagging_project.git
