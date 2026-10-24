import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder


def train_model(model, device, loss_fn, optimizer, scheduler,
                train_data_loader, test_data_loader,
                n_epochs=200):
    model = model.to(device)
    model.train()
    train_losses_avg_hist = []
    accuracy_hist = []
    best_acc = -np.inf
    best_model = None
    for epoch in range(n_epochs):
        train_losses = []
        for X_batch, y_batch in train_data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        curr_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        updated_lr = optimizer.param_groups[0]['lr']
        curr_train_losses_avg = sum(train_losses) / len(train_losses)
        train_losses_avg_hist.append(curr_train_losses_avg)
        for X_test, y_test in test_data_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            model.eval()
            y_pred = model(X_test)
            threshold = 0.5
            acc = ((y_pred > threshold).float() == y_test).float().mean()
            if acc > best_acc:
                best_acc = acc
                best_model = model.state_dict()
            print(f'Epoch {epoch + 1}/{n_epochs}, LR: {curr_lr:.6f} -> {updated_lr:.6f}, Loss: {curr_train_losses_avg:.4f}, Accuracy: {acc:.4f}')
            accuracy_hist.append(acc)
        print(f'Test Best Accuracy: {best_acc:.4f}')
    if best_model is not None:
        model.load_state_dict(best_model)
    return model, train_losses_avg_hist, accuracy_hist

class SonarDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1) # Reshape y to be a column vector

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        features = self.X[index]
        target = self.y[index]
        return features, target


if __name__ == "__main__":
    # Read data and convert to NumPy arrays
    data = pd.read_csv("./datasets/sonar.csv", header=None)
    X = data.iloc[:, :60].values
    y = data.iloc[:, 60].values

    # Encode class labels as integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Create DataLoader defining dataset as list of tuples (X, y)
    # X = torch.tensor(X, dtype=torch.float32)
    # y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)  # Reshape y to be a column vector
    # dataset = list(zip(X, y))
    dataset = SonarDataset(X, y)
    train_set, test_set = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set, shuffle=True, batch_size=16)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=len(test_set))

    model = nn.Sequential(
        nn.Linear(60, 60),
        nn.ReLU(),
        nn.Linear(60, 30),
        nn.ReLU(),
        nn.Linear(30, 1),
        nn.Sigmoid()
    )
    n_epochs = 200
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)

    best_model, train_losses_avg_hist, accuracy_hist = train_model(model, device='cpu', loss_fn=loss_fn,
                                                              optimizer=optimizer, scheduler=scheduler,
                                                              train_data_loader=train_loader,
                                                              test_data_loader=test_loader)

    with torch.no_grad():
        best_model.eval()
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to('cpu'), y_test.to('cpu')
            y_pred = model(X_test)
            loss = loss_fn(y_pred, y_test)
            threshold = 0.5
            acc = ((y_pred > threshold) == y_test).float().mean()
            print(f"Best Model Test Loss: {loss.item():.4f}, Best Model Test Accuracy: {acc:.4f}")

    plt.figure(figsize=(12, 5))
    plt.plot(train_losses_avg_hist, label='Train Loss')
    plt.plot(accuracy_hist, label='Accuracy')
    plt.title('Training Loss and Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    plt.show()

