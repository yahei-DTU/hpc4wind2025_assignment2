import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import product
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import time

# train_df = pd.read_csv("assignment 2/train_data.csv")
# test_df = pd.read_csv("assignment 2/test_data.csv")
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

new_y_train = train_df["total_load_actual"]
new_y_train = np.expand_dims(new_y_train, axis=1)
new_X_train = train_df.drop("total_load_actual", axis=1)

new_y_test = test_df["total_load_actual"]
new_y_test = np.expand_dims(new_y_test, axis=1)
new_X_test = test_df.drop("total_load_actual", axis=1)

# FCN model
class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()
        )
        self.fch = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()
            ) for _ in range(N_LAYERS - 1)
        ])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

# Evaluation Metrics
def compute_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

# Training function
def train_model(N_HIDDEN, N_LAYERS, learning_rate, batch_size, device, X_train, y_train, X_val, y_val, num_epochs=1000):
    model = FCN(X_train.shape[1], y_train.shape[1], N_HIDDEN, N_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    epoch_time = []

    for epoch in range(num_epochs):
        start_epoch = time.perf_counter()
        model.train()
        for xb, yb in dataloader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = nn.MSELoss()(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = nn.MSELoss()(val_preds, y_val)
            metrics = compute_metrics(y_val, val_preds)

        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch}: Train Loss = {loss.item():.6f}, Validation Loss = {val_loss.item():.6f}")
        #     print(f" -> MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}, R2: {metrics['r2']:.4f}\n")
        epoch_time.append(time.perf_counter() - start_epoch)

    return metrics, np.mean(epoch_time), np.std(epoch_time)

# Grid Search Runner
def run_grid_search(X_train, y_train, X_val, y_val, device):
    hidden_sizes = [32, 64]
    num_layers_list = [2, 3, 5]
    learning_rates = [1e-4, 1e-5]
    batch_sizes = [16, 32]
    num_epochs = 500

    grid_results = []
    best_by_metric = {
        'mse': {'score': float('inf'), 'params': None},
        'mae': {'score': float('inf'), 'params': None},
        'r2': {'score': -float('inf'), 'params': None}
    }

    for hs, nl, lr, bs in product(hidden_sizes, num_layers_list, learning_rates, batch_sizes):
        print(f"-> H={hs}, L={nl}, LR={lr}, BS={bs}")
        metrics, epoch_mean, epoch_std = train_model(hs, nl, lr, bs, device, X_train, y_train, X_val, y_val, num_epochs)

        row = {
            'mse': metrics['mse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'hidden_size': hs,
            'num_layers': nl,
            'learning_rate': lr,
            'batch_size': bs,
            'epoch_mean': epoch_mean,
            'epoch_std': epoch_std
        }
        
        grid_results.append(row)

        if metrics['mse'] < best_by_metric['mse']['score']:
            best_by_metric['mse'] = {'score': metrics['mse'], 'hs': hs, 'nl': nl,
                                    'lr': lr, 'bs': bs, 'em': epoch_mean, 'estd': epoch_std}
        if metrics['mae'] < best_by_metric['mae']['score']:
            best_by_metric['mae'] = {'score': metrics['mae'], 'hs': hs, 'nl': nl,
                                    'lr': lr, 'bs': bs, 'em': epoch_mean, 'estd': epoch_std}
        if metrics['r2'] > best_by_metric['r2']['score']:
            best_by_metric['r2'] = {'score': metrics['r2'], 'hs': hs, 'nl': nl,
                                    'lr': lr, 'bs': bs, 'em': epoch_mean, 'estd': epoch_std}

    return pd.DataFrame(grid_results), best_by_metric

# Main wrapper
def run_full_pipeline():
    # Set device
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

    # Fix random seeds
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)

    # Data prep
    X_train = torch.tensor(new_X_train.to_numpy(), dtype=torch.float32).to(device)
    y_train = torch.tensor(new_y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(new_X_test.to_numpy(), dtype=torch.float32).to(device)
    y_test = torch.tensor(new_y_test, dtype=torch.float32).to(device)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        start = time.perf_counter()
        torch.cuda.synchronize()
        results_df, bests = run_grid_search(X_train, y_train, X_test, y_test, device)
        torch.cuda.synchronize()
        end = time.perf_counter()
        torch.cuda.synchronize()
    else:
        start = time.perf_counter()
        results_df, bests = run_grid_search(X_train, y_train, X_test, y_test, device)
        end = time.perf_counter()

    total_time = end - start
    return results_df, bests, total_time, device

results, bests, time, device = run_full_pipeline()
print(f"\n{device} grid search time: {time:.2f} seconds")

print('Metric: MSE')
print(bests['mse'])

print('Metric: MAE')
print(bests['mae'])

print('Metric: R2')
print(bests['r2'])