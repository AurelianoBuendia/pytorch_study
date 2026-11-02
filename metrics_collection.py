
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                                        random_state=42, shuffle=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    model = nn.Sequential(
        nn.Linear(8, 24),
        nn.ReLU(),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1)
    )

    loss_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 100
    batch_size = 32
    batch_start = torch.arange(0, len(X_train), batch_size)
    train_mse_history = []
    test_mse_history = []
    test_mae_history = []
    for epoch in range(n_epochs):
        model.train()
        epoch_mse = []
        for start in batch_start:
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            epoch_mse.append(float(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_mse_history.append(sum(epoch_mse) / len(epoch_mse))
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            mse = loss_fn(y_pred, y_test)
            mae = mae_fn(y_pred, y_test)
            test_mse_history.append(float(mse.item()))
            test_mae_history.append(float(mae.item()))

    plt.figure(figsize=(12, 5))
    plt.plot(np.sqrt(train_mse_history), label="Train RMSE")
    plt.plot(np.sqrt(test_mse_history), label="Test RMSE")
    plt.plot(test_mae_history, label="Test MAE")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
