import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import read_files_from_folder, evaluate_and_write_results

# Tokenization and word embeddings with PyTorch
class TextDataset(Dataset):
    def __init__(self, texts, labels, vectorizer):
        self.texts = texts
        self.labels = labels  # Keep labels as NumPy array
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        vectorized_text = self.vectorizer.transform([text]).toarray().squeeze()
        return torch.FloatTensor(vectorized_text), torch.LongTensor([label])

class TextClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TextClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

data_folder_path = "../data/bbc"
texts, labels = read_files_from_folder(data_folder_path)

df = pd.DataFrame({'text': texts, 'label': labels})

df['text'] = df['text'].str.lower()

# Create label mapping
label_mapping = {'sport': 0, 'entertainment': 1, 'tech': 2, 'business': 3, 'politics': 4}


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Reset index of train and test dataframes
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

train_df['label'] = train_df['label'].map(label_mapping)
test_df['label'] = test_df['label'].map(label_mapping)

# Use CountVectorizer for simple word embeddings
vectorizer = CountVectorizer(stop_words='english')

# Fit and transform on the training data
X_train = vectorizer.fit_transform(train_df['text']).toarray()
y_train = train_df['label']

# Transform the testing data
X_test = vectorizer.transform(test_df['text']).toarray()
y_test = test_df['label']

# Create PyTorch datasets and dataloaders
train_dataset = TextDataset(train_df['text'], train_df['label'].to_numpy(), vectorizer)
test_dataset = TextDataset(test_df['text'], test_df['label'].to_numpy(), vectorizer)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


input_size = X_train.shape[1]
num_classes = len(set(labels))
model = TextClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()

model.eval()
all_predictions = []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Testing'):
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.tolist())

evaluate_and_write_results(test_df, all_predictions, "../results/torch_results_test.txt")