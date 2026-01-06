import os, re, gc, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    get_cosine_schedule_with_warmup
)

from torch.optim import AdamW


class CFG:
    NLI_MODEL = "../Models/nli-deberta-v3-base"
    EMB_MPNet = "../Models/all-mpnet-base-v2"
    EMB_GTE   = "../Models/gte-large-en-v1.5"

    MAX_LEN = 192
    BATCH = 16
    EPOCHS = 3
    LR = 2e-5
    FOLDS = 5

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    ALPHA = 0.70   
    BETA  = 0.15   
    GAMMA = 0.15   


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all(CFG.SEED)

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class NLIDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.pairs = []
        self.labels = []

        for t, y in zip(df.text, df.target):
            self.pairs.append((
                clean(t),
                "This tweet describes a real natural disaster or emergency."
            ))
            self.labels.append(2 if y == 1 else 0)  # entail / contradict

        self.tokenizer = tokenizer

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        p, h = self.pairs[idx]
        enc = self.tokenizer(
            p, h,
            max_length=CFG.MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


@torch.no_grad()
def compute_embedding_scores(model_path, texts, proto_pos, proto_neg):
    tok = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    enc_model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True
    ).to(CFG.DEVICE)
    enc_model.eval()

    def embed(sent):
        t = tok(sent, return_tensors="pt", truncation=True).to(CFG.DEVICE)
        out = enc_model(**t).last_hidden_state[:, 0]
        return out / out.norm(dim=1, keepdim=True)

    proto_pos_vec = embed(proto_pos)
    proto_neg_vec = embed(proto_neg)

    scores = []
    for t in tqdm(texts, desc=f"Embedding {os.path.basename(model_path)}"):
        v = embed(t)
        s = (v @ proto_pos_vec.T - 0.5 * v @ proto_neg_vec.T).item()
        scores.append(s)

    return np.array(scores)



def train_fold(fold, tr_df, va_df):
    tok = AutoTokenizer.from_pretrained(CFG.NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG.NLI_MODEL, num_labels=3
    ).to(CFG.DEVICE)

    tr_ds = NLIDataset(tr_df, tok)
    va_ds = NLIDataset(va_df, tok)

    tr_dl = DataLoader(tr_ds, CFG.BATCH, shuffle=True)
    va_dl = DataLoader(va_ds, CFG.BATCH)

    opt = AdamW(model.parameters(), lr=CFG.LR)
    steps = len(tr_dl) * CFG.EPOCHS
    sch = get_cosine_schedule_with_warmup(opt, steps // 10, steps)

    best = 0

    for ep in range(CFG.EPOCHS):
        model.train()
        for b in tr_dl:
            b = {k: v.to(CFG.DEVICE) for k, v in b.items()}
            loss = model(**b).loss
            loss.backward()
            opt.step()
            sch.step()
            opt.zero_grad()

        model.eval()
        probs, ys = [], []
        with torch.no_grad():
            for b in va_dl:
                b = {k: v.to(CFG.DEVICE) for k, v in b.items()}
                logit = model(**b).logits
                p = torch.softmax(logit, -1)[:, 2]
                probs.append(p.cpu())
                ys.append((b["labels"] == 2).int().cpu())

        probs = torch.cat(probs).numpy()
        ys = torch.cat(ys).numpy()
        f1 = f1_score(ys, probs > 0.5)

        if f1 > best:
            best = f1
            torch.save(model.state_dict(), f"nli_fold{fold}.pt")

        print(f"Fold {fold} Epoch {ep} F1={f1:.4f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

def main():
    train = pd.read_csv("train.csv")
    test  = pd.read_csv("test.csv")
    
    skf = StratifiedKFold(CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
    
    train_texts = [clean(t) for t in train.text]
    test_texts = [clean(t) for t in test.text]
    all_texts = train_texts + test_texts 
    
    for f, (tr, va) in enumerate(skf.split(train, train.target)):
        train_fold(f, train.iloc[tr], train.iloc[va])
    
    tok = AutoTokenizer.from_pretrained(CFG.NLI_MODEL)
    
    H_DIS = "This tweet describes a real natural disaster or emergency."
    H_JOK = "This tweet is a joke, exaggeration, or metaphor, not a real disaster."
    
    def nli_inference(texts):
        """对给定的文本列表进行NLI推理"""
        nli_dis = np.zeros(len(texts))
        nli_jok = np.zeros(len(texts))
        
        for f in range(CFG.FOLDS):
            model = AutoModelForSequenceClassification.from_pretrained(
                CFG.NLI_MODEL, num_labels=3
            ).to(CFG.DEVICE)
            model.load_state_dict(torch.load(f"nli_fold{f}.pt"))
            model.eval()
            
            with torch.no_grad():
                for i, t in tqdm(enumerate(texts), total=len(texts), desc=f"NLI Fold {f}"):
                    enc_d = tok(t, H_DIS, return_tensors="pt", truncation=True, max_length=CFG.MAX_LEN).to(CFG.DEVICE)
                    enc_j = tok(t, H_JOK, return_tensors="pt", truncation=True, max_length=CFG.MAX_LEN).to(CFG.DEVICE)
                    
                    p_d = torch.softmax(model(**enc_d).logits, -1)[0, 2]
                    p_j = torch.softmax(model(**enc_j).logits, -1)[0, 2]
                    
                    nli_dis[i] += p_d.item()
                    nli_jok[i] += p_j.item()
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
        
        nli_dis /= CFG.FOLDS
        nli_jok /= CFG.FOLDS
        return np.clip(nli_dis - 0.6 * nli_jok, 0, 1)
    
    nli_train = nli_inference(train_texts)
    nli_test = nli_inference(test_texts)
    
    
    proto_pos = "A real natural disaster or emergency is happening."
    proto_neg = "This is a joke or metaphor, not a real disaster."
    
    mpnet_scores = compute_embedding_scores(CFG.EMB_MPNet, all_texts, proto_pos, proto_neg)
    mpnet_train = mpnet_scores[:len(train)]
    mpnet_test = mpnet_scores[len(train):]
    
    gte_scores = compute_embedding_scores(CFG.EMB_GTE, all_texts, proto_pos, proto_neg)
    gte_train = gte_scores[:len(train)]
    gte_test = gte_scores[len(train):]
    

    for arr in [mpnet_train, mpnet_test, gte_train, gte_test]:
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr[:] = (arr - arr_min) / (arr_max - arr_min)
    
    def get_joke_flags(texts):
        JOKE_RE = re.compile(r"\b(lol|lmao|haha|😂|🤣|jk)\b", re.I)
        return np.array([1 if JOKE_RE.search(t) else 0 for t in texts])
    
    joke_train = get_joke_flags(train_texts)
    joke_test = get_joke_flags(test_texts)
    

    final_train = (
        CFG.ALPHA * nli_train +
        CFG.BETA  * mpnet_train +
        CFG.GAMMA * gte_train
    ) * (1 - 0.15 * joke_train)
    
    best_t, best_f1 = 0.5, 0
    for t in np.linspace(0.3, 0.7, 81): 
        f1 = f1_score(train.target, final_train > t)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    
    print(f"Best Threshold = {best_t:.3f}, F1 = {best_f1:.4f}")
    

    final_test = (
        CFG.ALPHA * nli_test +
        CFG.BETA  * mpnet_test +
        CFG.GAMMA * gte_test
    ) * (1 - 0.15 * joke_test)
    
    sub = pd.DataFrame({
        "id": test.id,
        "target": (final_test > best_t).astype(int)
    })
    sub.to_csv("submission6.csv", index=False)
    
if __name__ == "__main__":
    main()
