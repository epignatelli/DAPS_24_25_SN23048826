import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 25
BATCH_SIZE = 32
OUT_STEPS = 30
LR = 1e-3

# -----------------------------
# Dataset
# -----------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, seq_len=30, out_steps=30, target_col="Close"):
        self.data = torch.tensor(data.values, dtype=torch.float32)
        self.seq_len = seq_len
        self.out_steps = out_steps
        self.target_idx = data.columns.get_loc(target_col)

    def __len__(self):
        return len(self.data) - self.seq_len - self.out_steps + 1

    def __getitem__(self, idx):
        X = self.data[idx: idx + self.seq_len, :]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.out_steps, self.target_idx]
        return X, y

# -----------------------------
# Model
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=OUT_STEPS):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)       
        out = out[:, -1, :]        
        out = self.relu(out)
        out = self.fc(out)         
        return out

# -----------------------------
# Training Loop
# -----------------------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(X), y).item()
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}/{epochs} "
              f"Train MSE={train_losses[-1]:.4f} "
              f"Val MSE={val_losses[-1]:.4f}")

    return train_losses, val_losses

# -----------------------------
# Evaluation & Plots
# -----------------------------
def evaluate_model(model, test_loader, df_name):
    model.eval()
    preds, trues = [], []
    criterion = nn.MSELoss()

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            output = model(X)
            preds.append(output.cpu())
            trues.append(y.cpu())

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    mse = criterion(torch.tensor(preds), torch.tensor(trues)).item()
    print(f"Test MSE for {df_name}: {mse:.4f}")

    os.makedirs("inference", exist_ok=True)

    # Plot first test sample
    plt.figure(figsize=(10,5))
    plt.plot(trues[0], label="True", marker="o")
    plt.plot(preds[0], label="Predicted", marker="x")
    plt.title(f"Prediction (first test sample) – {df_name}")
    plt.xlabel("Day")
    plt.ylabel("Close Price")
    plt.legend()
    plt.savefig(f"inference/pred_sample_{df_name}.png")
    plt.close()

    # Plot all test predictions flattened
    plt.figure(figsize=(12,5))
    plt.plot(trues.flatten(), label="True", alpha=0.7)
    plt.plot(preds.flatten(), label="Predicted", alpha=0.7)
    plt.title(f"Predicted vs True (flattened test set) – {df_name}")
    plt.xlabel("Time step")
    plt.ylabel("Close Price")
    plt.legend()
    plt.savefig(f"inference/pred_all_{df_name}.png")
    plt.close()

    return mse

# -----------------------------
# Ablation Studies
# -----------------------------
def inference(dataframe: dict):
    stocks_data = dataframe['stock']
    covid_data = dataframe['covid']
    interest_rate_data = dataframe['interest']

    integrated_int = stocks_data.merge(interest_rate_data, left_index=True, right_index=True)
    integrated_full = integrated_int.merge(covid_data, left_index=True, right_index=True)

    ablations = {
        "stocks": stocks_data,
        "stocks_interest": integrated_int,
        "stocks_interest_covid": integrated_full
    }

    results = {}
    for name, df in ablations.items():
        print(f"\nRunning ablation: {name}")
        # chronological split
        n = len(df)
        train_df = df.iloc[:int(0.7*n)]
        val_df   = df.iloc[int(0.7*n):int(0.9*n)]
        test_df  = df.iloc[int(0.9*n):]

        # datasets
        train_ds = TimeSeriesDataset(train_df)
        val_ds   = TimeSeriesDataset(val_df)
        test_ds  = TimeSeriesDataset(test_df)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # model
        model = LSTMModel(input_size=df.shape[1]).to(DEVICE)

        # train
        train_losses, val_losses = train_model(model, train_loader, val_loader)

        # learning curve
        plt.figure()
        plt.plot(train_losses, label="Train MSE")
        plt.plot(val_losses, label="Val MSE")
        plt.title(f"Learning Curve – {name}")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.grid()
        os.makedirs("inference", exist_ok=True)
        plt.savefig(f"inference/learning_curve_{name}.png")
        plt.close()

        # evaluate
        mse = evaluate_model(model, test_loader, name)
        results[name] = mse

    return results