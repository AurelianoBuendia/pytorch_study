
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FunctionTransformer
from skorch import NeuralNetBinaryClassifier


class SonarClassifier(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.layers = []
        self.acts = []
        for i in range(n_layers):
            self.layers.append((nn.Linear(60, 60)))
            self.acts.append(nn.ReLU())
            self.add_module(f"layer{i}", self.layers[-1])
            self.add_module(f"act{i}", self.acts[-1])
        self.output = nn.Linear(60, 1)

    def forward(self, x):
        for layer, act in zip(self.layers, self.acts):
            x = act(layer(x))
        x = self.output(x)
        return x


if __name__ == '__main__':
    data = pd.read_csv("./datasets/sonar.csv", header=None)
    X = data.iloc[:, :60]
    y = data.iloc[:, -1]
    print(type(X))
    print(type(y))

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    model = NeuralNetBinaryClassifier(
        module=SonarClassifier,
        criterion=nn.BCEWithLogitsLoss,
        optimizer=optim.Adam,
        lr=0.0001,
        max_epochs=150,
        batch_size=10
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('float32', FunctionTransformer(func=lambda X: torch.tensor(X, dtype=torch.float32))),
        ('sonarmodel', model.initialize()),
        ])

    param_grid = {
        'sonarmodel__module__n_layers': [1, 3, 5],
        'sonarmodel__lr': [0.0001, 0.001, 0.01, 0.1],
        'sonarmodel__max_epochs': [100, 150, 200]
    }

    # kfold = StratifiedKFold(n_splits=5, shuffle=True)
    # results = cross_val_score(model, X, y, cv=kfold)
    grid_search = GridSearchCV(pipe, param_grid, scoring='accuracy', verbose=1, cv=3)
    result = grid_search.fit(X, y)
    print(f"Best score: {result.best_score_} using {result.best_params_}")
    means = result.cv_results_['mean_test_score']
    stds = result.cv_results_['std_test_score']
    params = result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f"Mean: {mean:.3f}, Std: {stdev:.3f} using {param}")



