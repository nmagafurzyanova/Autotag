import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def read_files_from_folder(folder_path):
    texts = []
    labels = []
    for category_label, category_folder in enumerate(os.listdir(folder_path)):
        category_path = os.path.join(folder_path, category_folder)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                if os.path.isfile(file_path) and file_path.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                        texts.append(text)
                        labels.append(category_folder)
    return texts, labels

def write_results_to_file(accuracy, classification_rep, file_path):
    with open(file_path, "w") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write("Classification Report:\n")
        file.write(classification_rep)

def evaluate_and_write_results(test_df, all_predictions, file_path):
    accuracy = accuracy_score(test_df['label'], all_predictions)
    classification_rep = classification_report(test_df['label'], all_predictions)
    
    write_results_to_file(accuracy, classification_rep, file_path)
