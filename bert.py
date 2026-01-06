import os
import shutil
import math
import json
import torch.nn as nn
import pandas as pd
import numpy as np
import re
from spellchecker import SpellChecker  
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import resample
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from transformers import get_scheduler
import warnings


warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


spell = SpellChecker()
sia = SentimentIntensityAnalyzer()

def clean_2(text):
    text = re.sub(r'\b(u|ur|urs)\b', 'you', text.lower())
    text = re.sub(r'\b(r|are)\b', 'are', text)
    text = re.sub(r'[:;]-?[()|D]', ' emoji ', text) 
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', ' number ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    corrected = [spell.correction(w) if spell.correction(w) else w for w in words]
    return ' '.join(corrected)

def get_sentiment_features(text):
    scores = sia.polarity_scores(text)
    return [scores['neg'], scores['neu'], scores['pos'], scores['compound']]

train_df['clean_text'] = train_df['text'].apply(clean_2)
test_df['clean_text'] = test_df['text'].apply(clean_2)

train_df['full_text'] = train_df['clean_text'] + ' ' + train_df['keyword'].fillna('').str.lower() + ' ' + train_df['location'].fillna('').str.lower()
test_df['full_text'] = test_df['clean_text'] + ' ' + test_df['keyword'].fillna('').str.lower() + ' ' + test_df['location'].fillna('').str.lower()

train_sentiments = np.array([get_sentiment_features(t) for t in train_df['text']])
test_sentiments = np.array([get_sentiment_features(t) for t in test_df['text']])

df_majority = train_df[train_df.target == 0]
df_minority = train_df[train_df.target == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)

def add_noise(text, noise_level=0.1):
    words = text.split()
    num_noise = int(len(words) * noise_level)
    for _ in range(num_noise):
        idx = np.random.randint(len(words))
        words[idx] = spell.candidates(words[idx]).pop() if spell.candidates(words[idx]) else words[idx]  
    return ' '.join(words)

df_minority_upsampled['full_text'] = df_minority_upsampled['full_text'].apply(add_noise)

train_df_balanced = pd.concat([df_majority, df_minority_upsampled])
train_sentiments_bal = np.concatenate([train_sentiments[train_df.target == 0], train_sentiments[train_df.target == 1]])  

train_texts, val_texts, train_sent, val_sent, train_labels, val_labels = train_test_split(
    train_df_balanced['full_text'], train_sentiments_bal, train_df_balanced['target'], 
    test_size=0.2, random_state=42, stratify=train_df_balanced['target']
)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class AdvancedDataset(Dataset):
    def __init__(self, texts, sentiments, labels=None):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128, return_tensors='pt')
        self.sentiments = torch.tensor(sentiments, dtype=torch.float)
        self.labels = torch.tensor(labels.values, dtype=torch.long) if labels is not None else None
    
    def __len__(self): return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['sentiments'] = self.sentiments[idx]
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item

train_dataset = AdvancedDataset(train_texts, train_sent, train_labels)
val_dataset = AdvancedDataset(val_texts, val_sent, val_labels)
test_dataset = AdvancedDataset(test_df['full_text'], test_sentiments)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

class AdvancedClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        self.mlp = nn.Sequential(
            nn.Linear(2 + 4, 128),  
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )
    
    def forward(self, input_ids, attention_mask, sentiments):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        combined = torch.cat([logits, sentiments], dim=1)
        return self.mlp(combined)

model = AdvancedClassifier().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 4
num_training_steps = num_epochs * len(train_loader)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

best_f1 = 0
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sentiments = batch['sentiments'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, sentiments)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiments = batch['sentiments'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask, sentiments)
            pred = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(labels.cpu().numpy())
    f1 = f1_score(trues, preds)
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), 'best_model.pt')  

print(f" F1: {best_f1:.4f}")

model.load_state_dict(torch.load('best_model.pt'))
model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sentiments = batch['sentiments'].to(device)
        outputs = model(input_ids, attention_mask, sentiments)
        pred = torch.argmax(outputs, dim=1).cpu().numpy()
        test_preds.extend(pred)

submission = pd.DataFrame({'id': test_df['id'], 'target': test_preds})
submission.to_csv('submission_3.csv', index=False)
