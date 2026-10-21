
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pprint
import pickle


#Load data into NumPy arrays
data = load_iris()
X, y = data["data"], data["target"]
#convert NumPy array into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
#split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

#PyTorch model
class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x)) # LogSoftmax for multi-class classification
        return x

model = Multiclass()

#loss metric and optimizer
loss_fn = nn.NLLLoss()  # Negative Log Likelihood Loss for multi-class classification. This is used in conjunction
                        # with LogSoftmax activation resulting a function similar to cross-entropy loss applied to
                        # a linear layer (no activation).
optimizer = optim.Adam(model.parameters(), lr=0.001)
#prepare model and training parameters
n_epochs = 100
batch_size = 5
batch_start = torch.arange(0, len(X), batch_size)
#training loop
for epoch in range(n_epochs):
    for start in batch_start:
        #take a batch
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        #forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        #update weights
        optimizer.step()

y_pred = model(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
print(f"Accuracy: {float(acc.detach().numpy()):.2f}")

#Save model
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(model.state_dict())
torch.save(model.state_dict(), "./output/iris-model.pth")

def load_model(model, path):
    with open(path, "rb") as f:
        model.load_state_dict(torch.load(f))
    return model


if __name__ == "__main__":
    # Load the model
    new_model = Multiclass()
    new_model = load_model(new_model, "./output/iris-model.pth")
    # Test the loaded model
    y_pred_loaded = new_model(X_test)
    acc_loaded = (torch.argmax(y_pred_loaded, 1) == y_test).float().mean()
    print(f"Accuracy of loaded model: {float(acc_loaded.detach().numpy()):.2f}")
