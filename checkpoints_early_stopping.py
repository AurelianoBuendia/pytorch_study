
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, default_collate
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder


def checkpoint(model, filename):
    """
    Save the model's state_dict to a file.

    Args:
        model (torch.nn.Module): The model to save.
        filename (str): The path to the file where the model will be saved.
    """
    torch.save({
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
    }, filename)

def resume(model, filename):
    """
    Load the model's state_dict from a file.

    Args:
        model (torch.nn.Module): The model to load the state_dict into.
        filename (str): The path to the file from which the model will be loaded.
    """
    checkpoint = torch.load(filename, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


if __name__ == '__main__':
    data = fetch_openml("electricity", version=1, parser="auto") # Binary classification dataset

    X = data["data"].astype(float).values
    y = data['target']
    print(type(X))
    print(type(y))

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    trainset, testset = random_split(TensorDataset(X, y), [0.7, 0.3])

    model = nn.Sequential(
        nn.Linear(8, 12),
        nn.ReLU(),
        nn.Linear(12, 12),
        nn.ReLU(),
        nn.Linear(12, 1),
        nn.Sigmoid()
    )

    n_epochs = 100
    batch_size = 32
    loader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    X_test, y_test = default_collate(testset)

    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    start_epoch = 0
    # Check if a checkpoint exists and resume training from it
    if start_epoch > 0:
        resume(model, f"./output/checkpoints/epoch-{start_epoch}.pth")
        print(f"Resumed training from epoch {start_epoch}")

    early_stop_threshold = 5
    best_accuracy = -1.0
    best_epoch = -1
    for epoch in range(start_epoch, n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        y_pred = model(X_test)
        acc = ((y_pred > 0.5) == y_test).float().mean()
        print(f"End of epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_epoch = epoch
            checkpoint(model, f"./output/checkpoints/best_model.pth")
        elif epoch - best_epoch >= early_stop_threshold:
            print(f"Early stopping triggered at epoch {epoch}. Best accuracy: {best_accuracy:.4f} at epoch {best_epoch}.")
            break

    resume(model, f"./output/checkpoints/best_model.pth")
