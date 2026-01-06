# 导入库
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.utils import resample
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('punkt')


np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df['text_full'] = train_df['text'].fillna('') + ' ' + train_df['keyword'].fillna('') + ' ' + train_df['location'].fillna('')
test_df['text_full'] = test_df['text'].fillna('') + ' ' + test_df['keyword'].fillna('') + ' ' + test_df['location'].fillna('')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

results_lr = []

configs = [
    ("A_no_clean_no_balance", False, False),
    ("B_clean_no_balance",   True,  False),
    ("C_no_clean_balance",   False, True),
    ("D_clean_balance",       True,  True)
]

for name, do_clean, do_balance in configs:
    print(f"\n=== {name} ===")
    
    train_text = train_df['text_full'].apply(clean_text) if do_clean else train_df['text_full']
    test_text = test_df['text_full'].apply(clean_text) if do_clean else test_df['text_full']
    
    df_train = pd.DataFrame({'text': train_text, 'target': train_df['target']})
    if do_balance:
        df_majority = df_train[df_train.target == 0]
        df_minority = df_train[df_train.target == 1]
        df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
        df_balanced = pd.concat([df_majority, df_minority_upsampled])
        X_texts = df_balanced['text']
        y = df_balanced['target']
    else:
        X_texts = df_train['text']
        y = df_train['target']
    
    X_train, X_val, y_train, y_val = train_test_split(X_texts, y, test_size=0.2, random_state=42, stratify=y)
    
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        stop_words='english' if do_clean else None
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(test_text)
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced' if not do_balance else None)
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_val_vec)
    f1 = f1_score(y_val, y_pred)
    print(f"F1: {f1:.4f}")
    
    test_pred = model.predict(X_test_vec)
    submission = pd.DataFrame({'id': test_df['id'], 'target': test_pred})
    filename = f"submission_lr_{name}.csv"
    submission.to_csv(filename, index=False)
    
    results_lr.append({'config': name, 'f1': f1, 'file': filename})