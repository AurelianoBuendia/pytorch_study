
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
import torchmetrics
import optuna


device = 'cuda' if torch.cuda.is_available() else 'cpu'
toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

train_and_valid_data = torchvision.datasets.FashionMNIST(root='datasets', train=True, download=False, transform=toTensor)
test_data = torchvision.datasets.FashionMNIST(root='datasets', train=False, download=False, transform=toTensor)

torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(train_and_valid_data, [55_000, 5_000])
train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

class ImageClassifier(nn.Module):
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes)
        )

    def forward(self, X):
        return self.mlp(X)

def train(model, device, optimizer, criterion, train_loader, n_epochs):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        mean_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{n_epochs}; Loss: {mean_loss:.4f}")

def train2(model, optimizer, criterion, metric,
           train_loader, valid_loader,
           n_epochs):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.
        metric.reset()
        for X_batch, y_batch in train_loader:
            model.train()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            evaluate_tm(model, device, valid_loader, metric).item())
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.4f}, "
              f"valid metric: {history['valid_metrics'][-1]:.4f}")
    return history

#torch.manual_seed(42)
#learning_rate = 0.01
#model = nn.Sequential(
#    nn.Linear(n_features, 50), nn.ReLU(),
#    nn.Linear(50, 40), nn.ReLU(),
#    nn.Linear(40, 30), nn.ReLU(),
#    nn.Linear(30, 1)
#)
#model = model.to(device)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
#mse = nn.MSELoss()
#rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
#history = train2(model, optimizer, mse, rmse, train_loader, valid_loader,
#                 n_epochs)

# Since we compute the training metric
#plt.plot(np.arange(n_epochs) + 0.5, history["train_metrics"], ".--",
#         label="Training")
#plt.plot(np.arange(n_epochs) + 1.0, history["valid_metrics"], ".-",
#         label="Validation")
#plt.xlabel("Epoch")
#plt.ylabel("RMSE")
#plt.grid()
#plt.title("Learning curves")
#plt.axis([0.5, 20, 0.4, 1.0])
#plt.legend()
#plt.show()

def evaluate(model, device, data_loader, metric_fn, aggregate_fn=torch.mean):
    model.eval()
    metrics = []
    with torch.no_grad():
        for X_batch, Y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric = metric_fn(y_pred, y_batch)
            metrics.append(metric)
    return aggregate_fn(torch.stack(metrics))
 
def evaluate_tm(model, device, data_loader, metric):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch) # Update metric at each iteration
    return metric.compute() # Compute the final result at the end

# torch.manual_seed(42)
# model = ImageClassifier(n_inputs=28*28, n_hidden1=300, n_hidden2=100, n_classes=10)
# xentropy = nn.CrossEntropyLoss()
#
# accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=10).to(device)
# learning_rate = 0.1
# n_epochs = 50
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# train(model, device, optimizer, xentropy, train_loader, n_epochs)
# evaluate_tm(model, device, valid_loader, accuracy)
# model.eval()
# X_new, y_new = next(iter(valid_loader))
# X_new = X_new[:3].to(device)
# with torch.no_grad():
#     y_pred_logits = model(X_new)
# print(y_pred_logits)
#
# print(type(y_pred_logits))
# y_pred = y_pred_logits.argmax(dim=1)
# print(type(y_pred))
# print(y_pred)
#
# print([train_and_valid_data.classes[index] for index in y_pred])
# y_proba = F.softmax(y_pred_logits, dim=1)
# y_proba.round(decimals=3)
#
# y_top4_logits, y_top4_indices = torch.topk(y_pred_logits, k=4, dim=1)
# y_top4_probas = F.softmax(y_top4_logits, dim=1)
# y_top4_probas.round(decimals=3)
# print(y_top4_indices)

def objective(n_epochs, trial, train_loader, valid_loader):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    n_hidden = trial.suggest_int("n_hidden", 20, 300)
    model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=n_hidden,
                            n_hidden2=n_hidden, n_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    xentropy = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    accuracy = accuracy.to(device)
    best_validation_accuracy = 0.0
    for epoch in range(n_epochs):
        history = train2(model, optimizer, xentropy, accuracy, train_loader,
                         valid_loader, n_epochs=1)
        validation_accuracy = max(history["valid_metrics"])
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
        trial.report(validation_accuracy, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best_validation_accuracy

# objective_with_data = lambda trial: objective(
#     trial, train_loader=train_loader, valid_loader=valid_loader)

# objective_with_data = partial(objective, train_loader=train_loader,
#                               valid_loader=valid_loader)
# torch.manual_seed(42)
# sampler = optuna.samplers.TPESampler(seed=42)
# pruner = optuna.pruners.MedianPruner()
# study = optuna.create_study(direction="maximize", sampler=sampler,
#                             pruner=pruner)
# study.optimize(objective_with_data, n_trials=20)

if __name__ == '__main__':
    x = torch.tensor( 3.6, requires_grad=True)
    y = x * x
    y.retain_grad()
    y.backward()
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x.grad_fn = {x.grad_fn}")
    print(f"x.grad = {x.grad}")
    print(f"x.grad_fn = {y.grad_fn}")
    print(f"x.grad = {y.grad}")

    input = torch.randn(32, 10, 5, 5)
    # With default parameters
    print(input.size())
    m = nn.Flatten()
    output = m(input)
    print(output.size())
    # With non-default parameters
    m = nn.Flatten(0, 2)
    output = m(input)
    print(output.size())

    m = nn.Dropout1d(p=0.2)
    input = torch.randn(10, 8)
    output = m(input)
    print(f"Input: {input}")
    print(f"Output: {output}")


