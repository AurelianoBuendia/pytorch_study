
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


class PimaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 12)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.output(x)
        x = self.act_out(x)
        return x


if __name__ == '__main__':
    dataset = pd.read_csv('./datasets/Pima_Indians_Diabetes/diabetes.csv')
    print(dataset.head())
    print(dataset.shape)
    dataset = dataset.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(dataset[:, :8], dataset[:, -1:], test_size=0.2, shuffle=True)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}\n")
    print(f"X_train: {X_train[:5]},\n y_train: {y_train[:5]}\n")
    print(f"X_test: {X_test[:5]},\n y_test: {y_test[:5]}\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_train = torch.tensor(X_train, dtype=torch.float32, requires_grad=True, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, requires_grad=True, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, requires_grad=True, device=device)

    n_epochs = 50
    batch_size = 32
    batches_per_epoch = len(X_train) // batch_size
    model = PimaClassifier()
    print(model)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss = []
    train_acc = []
    test_acc = []
    threshold = 0.5
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tgrange(batches_per_epoch, unit="batch", mininterval=0) as bar:
            bar.set_description(f"Epoch {epoch + 1}/{n_epochs}")
            for i in bar:
                start = i * batch_size
                end = start + batch_size
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                accuracy = ((y_pred > threshold) == y_batch).float().mean()
                train_loss.append(loss.item())
                train_acc.append(accuracy.item())
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar.set_postfix(
                    loss=float(loss),
                    acc=f"{float(accuracy) * 100:.2f}%"
                )
        # Evaluate on test set at the end of each epoch
        with torch.no_grad():
            model.eval()
            y_pred = model(X_test)
            y_pred_classes = (y_pred > threshold).int()
            test_accuracy = (y_pred_classes == y_test).float().mean()
            test_acc.append(test_accuracy.item())
            print(f"Test Accuracy after epoch {epoch + 1}: {test_accuracy.item() * 100:.2f}%")
    # Final evaluation on test set
    with torch.no_grad():
        model.eval()
        y_pred = model(X_test)
        y_pred_classes = (y_pred > threshold).int()
        accuracy = (y_pred_classes == y_test).float().mean()
        print(f"Final Test Accuracy: {accuracy.item() * 100:.2f}%")
    # Plotting the training loss and accuracy
    avg_train_acc = []
    for i in range(n_epochs):
        start = i * batches_per_epoch
        end = start + batches_per_epoch
        average = sum(train_acc[start:end]) / batches_per_epoch
        avg_train_acc.append(average)
    plt.figure(figsize=(12, 5))
    plt.plot(avg_train_acc, label="train accuracy")
    plt.plot(test_acc, label="test accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.ylim(0)
    plt.show()


