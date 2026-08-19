import os
import sys

from os.path import join

import torch
import pickle

import argparse
parser = argparse.ArgumentParser(description='Run ID for the experiment')
parser.add_argument('--run_id', type=str, required=True, help='Run ID for the experiment')
parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
args = parser.parse_args()

print("[Info] Dataset loading...")

train_emb = torch.load("../../embeddings/train_embedding.pt",  weights_only=True)
val_emb = torch.load("../../embeddings/val_embedding.pt",  weights_only=True)

print("[Info] NumPy conversion...")
X_train = train_emb["X"].cpu().numpy()  
y_train = train_emb["y_true"].squeeze().cpu().numpy().astype(int)

X_val = val_emb["X"].numpy()
y_val = val_emb["y_true"].squeeze().cpu().numpy().astype(int)

from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(
    loss="log",          # logistic regression
    penalty="l2",             # same as CellTypist (ridge regularization)
    alpha=1e-4,               # regularization strength (tune if needed)
    max_iter=1000,            # max number of passes over data
    tol=1e-3,                 # stop when convergence is good enough
    random_state=args.seed
)

print("[Info] Training logistic regression classifier...")
clf.fit(X_train, y_train)

save_dir = f'../checkpoints/LogisticRegression/run_{args.run_id}/'
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, 'logistic_regression_clf.pkl') 

with open(save_path, 'wb') as f:
    pickle.dump(clf, f)

print("[Info] Training complete")

