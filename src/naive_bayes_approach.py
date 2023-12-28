import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils import read_files_from_folder, evaluate_and_write_results

data_folder_path = "../data/bbc"

texts, labels = read_files_from_folder(data_folder_path)

df = pd.DataFrame({'text': texts, 'label': labels})

df['text'] = df['text'].str.lower()

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Use CountVectorizer for simple word embeddings
vectorizer = CountVectorizer(stop_words='english')

# Fit and transform on the training data
X_train = vectorizer.fit_transform(train_df['text']).toarray()
y_train = train_df['label']

# Transform the testing data
X_test = vectorizer.transform(test_df['text']).toarray()
y_test = test_df['label']

# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model

evaluate_and_write_results(test_df, y_pred, "../results/naive_bayes_results_test.txt")