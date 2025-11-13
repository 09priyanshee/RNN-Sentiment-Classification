"""
Run sentiment classification experiments using PyTorch.
Each configuration tests a combination of model type, optimizer, and sequence length.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score


# --- Model Definitions ---
# The original LSTMClassifier is now the base for the recurrent models.

class RNNClassifier(nn.Module):
    """Simple RNN-based sentiment classifier"""
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=128, output_dim=1, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Use simple RNN
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        
        # Linear layer expects hidden_dim * num_directions
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        # RNN outputs: (output, hidden)
        _, hidden = self.rnn(embedded)
        
        # Hidden shape: (num_layers * num_directions, batch_size, hidden_size)
        # We take the last layer's hidden state.
        # If bidirectional, we concatenate the forward and backward hidden states
        if self.num_directions == 2:
            # Concatenate (forward, backward) from the last layer
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # Take the hidden state of the last layer
            hidden = hidden[-1, :, :]
            
        out = self.fc(hidden)
        return self.sigmoid(out)


class LSTMClassifier(nn.Module):
    """LSTM-based sentiment classifier (Unidirectional)"""
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=128, output_dim=1, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # LSTM outputs: (output, (h_n, c_n))
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        
        # Linear layer expects hidden_dim * num_directions
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        # LSTM outputs: (output, (h_n, c_n))
        _, (hidden, _) = self.lstm(embedded)
        
        # Hidden shape: (num_layers * num_directions, batch_size, hidden_size)
        if self.num_directions == 2:
            # Concatenate (forward, backward) from the last layer
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # Take the hidden state of the last layer
            hidden = hidden[-1, :, :]
            
        out = self.fc(hidden)
        return self.sigmoid(out)


# --- Training Function ---
def train_model(model, train_loader, val_loader, epochs=3, lr=0.001, optimizer_name="Adam", clip_grad_norm=None, device="cpu"):
    criterion = nn.BCELoss()
    model.to(device)

    # Initialize Optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        # Using a higher LR is common for SGD
        optimizer = optim.SGD(model.parameters(), lr=0.1 if lr == 0.001 else lr, momentum=0.9)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X, y = X.to(device), y.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            
            # Gradient Clipping
            if clip_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        # Validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                outputs = model(X)
                preds = (outputs.cpu().numpy() > 0.5).astype(int)
                y_true.extend(y.numpy())
                y_pred.extend(preds)
        acc = accuracy_score(y_true, y_pred)
        history["val_acc"].append(acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

    return history


# --- Experiment Runner ---
def run_all_experiments(n_epochs=10):
    print("\n=== Running all experiments ===")
    os.makedirs("results", exist_ok=True)
    
    # --- Experiment Grid ---
    lengths = [25, 50, 100]
    optimizers = ["Adam", "SGD", "RMSprop"]
    
    # Architecture and Bi-directional status
    architectures = [
        ("RNN", False, RNNClassifier),
        ("LSTM", False, LSTMClassifier),
        ("Bi-LSTM", True, LSTMClassifier), # Bi-LSTM is implemented by setting bidirectional=True in LSTMClassifier
    ]
    
    # Stability: No strategy (None) vs. Gradient Clipping (clip_val=1.0)
    stability_strategies = [
        ("None", None),
        ("Clipping", 1.0)
    ]
    
    # Full Experiment List
    experiment_list = []
    for arch_name, is_bidirectional, ModelClass in architectures:
        for opt_name in optimizers:
            for clip_name, clip_val in stability_strategies:
                for max_len in lengths:
                    experiment_list.append({
                        "architecture": arch_name,
                        "bidirectional": is_bidirectional,
                        "ModelClass": ModelClass,
                        "optimizer": opt_name,
                        "clip_strategy": clip_name,
                        "clip_val": clip_val,
                        "length": max_len,
                    })
                    
    results = []
    
    for i, exp in enumerate(experiment_list):
        print("-" * 50)
        print(f"EXPERIMENT {i+1}/{len(experiment_list)}:")
        print(f"  Model: {exp['architecture']} (Bi: {exp['bidirectional']})")
        print(f"  Optimizer: {exp['optimizer']}")
        print(f"  Length: {exp['length']}")
        print(f"  Stability: {exp['clip_strategy']}")
        print("-" * 50)

        # Load data
        try:
            with open(f"data/preprocessed_len{exp['length']}.pkl", "rb") as f:
                data = pickle.load(f)
        except FileNotFoundError:
            print(f"ERROR: Data for length {exp['length']} not found. Skipping.")
            continue

        X_train = torch.tensor(np.array(data['X_train']))
        y_train = torch.tensor(np.array(data['y_train']))
        X_test = torch.tensor(np.array(data['X_test']))
        y_test = torch.tensor(np.array(data['y_test']))

        # Dataloaders
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

        # Model initialization
        model = exp["ModelClass"](
            vocab_size=data.get("num_words", 10000), 
            bidirectional=exp["bidirectional"]
        )
        
        # Training
        history = train_model(
            model, 
            train_loader, 
            val_loader, 
            epochs=n_epochs, 
            optimizer_name=exp["optimizer"], 
            clip_grad_norm=exp["clip_val"]
        )

        # Save results
        acc = history["val_acc"][-1]
        
        results.append({
            "architecture": exp["architecture"],
            "optimizer": exp["optimizer"],
            "clip_strategy": exp["clip_strategy"],
            "length": exp["length"],
            "accuracy": acc,
        })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("results/metrics.csv", index=False)
    print("\n✓ Experiments completed. Results saved to results/metrics.csv")
    
    # Save all history for detailed analysis/plots (optional but good practice)
    # Note: Full history saving would require a more complex structure, 
    # but the metrics CSV now holds the required final results.


# --- Script Entry Point ---
if __name__ == "__main__":
    # The run_all.py script will call this with full=True (or similar), so 
    # we'll keep the full run as the default for now.
    run_all_experiments(n_epochs=10) # Set to 10 epochs as per your initial run