import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import resample
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_df = pd.read_csv('train.csv')
test_df  = pd.read_csv('test.csv')

train_df['text_full'] = (
    train_df['text'].fillna('') + ' ' +
    train_df['keyword'].fillna('') + ' ' +
    train_df['location'].fillna('')
)
test_df['text_full'] = (
    test_df['text'].fillna('') + ' ' +
    test_df['keyword'].fillna('') + ' ' +
    test_df['location'].fillna('')
)

stop_words = set(stopwords.words('english'))

def clean_text_to_tokens(text: str):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return tokens

def build_vocab(token_lists, min_freq=2, max_size=10000):
    counter = Counter()
    for toks in token_lists:
        counter.update(toks)

    vocab = {'<pad>': 0, '<unk>': 1}
    idx = 2
    for w, freq in counter.most_common(max_size):
        if freq >= min_freq:
            vocab[w] = idx
            idx += 1
    return vocab

class TweetDataset(Dataset):
    def __init__(self, token_lists, labels=None, vocab=None, max_len=60):
        self.token_lists = token_lists          
        self.labels      = labels               
        self.vocab       = vocab
        self.max_len     = max_len

    def __len__(self):
        return len(self.token_lists)

    def __getitem__(self, idx):
        tokens = self.token_lists[idx]
        indices = [self.vocab.get(t, self.vocab['<unk>']) for t in tokens[:self.max_len]]
        pad_id = self.vocab['<pad>']
        if len(indices) < self.max_len:
            indices += [pad_id] * (self.max_len - len(indices))
        x = torch.tensor(indices, dtype=torch.long)

        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        return x


def collate_fn(batch):
    if isinstance(batch[0], tuple):          
        xs, ys = zip(*batch)
        xs = pad_sequence(xs, batch_first=True, padding_value=0)
        ys = torch.stack(ys)
        return xs, ys
    else:                                    
        xs = pad_sequence(batch, batch_first=True, padding_value=0)
        return xs


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        x = self.dropout(h_n[-1])
        return self.fc(x)


configs = [
    ("A_no_clean_no_balance", False, False),
    ("B_clean_no_balance",   True,  False),
    ("C_no_clean_balance",   False, True),
    ("D_clean_balance",       True,  True)
]

results = []

for cfg_name, do_clean, do_balance in configs:
    print(f"\n{'='*25} {cfg_name} {'='*25}")

    if do_clean:
        train_tokens = train_df['text_full'].apply(clean_text_to_tokens).tolist()
        test_tokens  = test_df['text_full'].apply(clean_text_to_tokens).tolist()
    else:
        train_tokens = [word_tokenize(t.lower()) for t in
                        train_df['text_full']]
        test_tokens  = [word_tokenize(t.lower()) for t in
                        test_df['text_full']]

    df_tmp = pd.DataFrame({
        'tokens': train_tokens,
        'target': train_df['target'].values
    })

    if do_balance:
        df_min = df_tmp[df_tmp.target == 1]
        df_maj = df_tmp[df_tmp.target == 0]
        df_min_up = resample(df_min, replace=True,
                             n_samples=len(df_maj), random_state=42)
        df_bal = pd.concat([df_maj, df_min_up])
        train_tok_list = df_bal['tokens'].tolist()
        train_lab_list = df_bal['target'].tolist()
    else:
        train_tok_list = df_tmp['tokens'].tolist()
        train_lab_list = df_tmp['target'].tolist()

    train_t, val_t, train_l, val_l = train_test_split(
        train_tok_list, train_lab_list,
        test_size=0.2, random_state=42, stratify=train_lab_list
    )

    vocab = build_vocab(train_t, min_freq=2, max_size=12000)
    vocab_size = len(vocab)
    print(f"vocab size: {vocab_size}")

    train_ds = TweetDataset(train_t, train_l, vocab)
    val_ds   = TweetDataset(val_t,   val_l,   vocab)
    test_ds  = TweetDataset(test_tokens, vocab=vocab)

    train_loader = DataLoader(train_ds, batch_size=64,
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=64,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=64,
                              collate_fn=collate_fn)

    model = LSTMClassifier(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_f1 = 0.0
    for epoch in range(6):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb)
                pred = out.argmax(1).cpu().numpy()
                preds.extend(pred)
                trues.extend(yb.numpy())
        f1 = f1_score(trues, preds)
        if f1 > best_f1:
            best_f1 = f1

    print(f"Best Val F1: {best_f1:.4f}")

    model.eval()
    test_preds = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            pred = out.argmax(1).cpu().numpy()
            test_preds.extend(pred)

    sub = pd.DataFrame({'id': test_df['id'], 'target': test_preds})
    fname = f"submission_lstm_{cfg_name}.csv"
    sub.to_csv(fname, index=False)

    results.append({'cfg': cfg_name, 'f1': best_f1, 'file': fname})


for r in results:
    print(f"{r['cfg']:<25} F1: {r['f1']:.4f}  →  {r['file']}")