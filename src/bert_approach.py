import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig
from utils import read_files_from_folder, evaluate_and_write_results

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
        }, self.labels[idx]

data_folder_path = "../data/bbc"
texts, labels = read_files_from_folder(data_folder_path)

df = pd.DataFrame({'text': texts, 'label': labels})
df['text'] = df['text'].str.lower()

label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
df['label'] = df['label'].map(label_mapping)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
num_labels_in_data = len(df['label'].unique())
config = BertConfig.from_pretrained('bert-base-uncased', num_labels=num_labels_in_data)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)

train_dataset = CustomDataset(train_df['text'], train_df['label'], tokenizer)
test_dataset = CustomDataset(test_df['text'], test_df['label'], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

num_labels_in_data = len(df['label'].unique())
num_output_units = model.config.num_labels

optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        inputs = batch[0]
        labels = batch[1].to(device)

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()


# Salvataggio del modello
save_folder = '../models'
os.makedirs(save_folder, exist_ok=True)

save_path = os.path.join(save_folder, 'bert_model.pth')

torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, save_path)


# Evaluation
model.eval()
all_predictions = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Testing'):
        inputs = batch[0]
        labels = batch[1].to(device)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        all_predictions.extend(predictions.cpu().numpy().tolist())

# Calcolo dell'accuratezza e stampa dei risultati
evaluate_and_write_results(test_df, all_predictions, "../results/bert_results_test.txt")
