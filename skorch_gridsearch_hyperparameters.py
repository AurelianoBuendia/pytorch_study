
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class PimaClassifier(nn.Module):
    def __init__(self, n_neurons=12):
        super().__init__()
        self.layer = nn.Linear(8, n_neurons)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.8)
        self.output = nn.Linear(n_neurons, 1)
        self.prob = nn.Sigmoid()
        self.weight_constraint = 2.0
        # Manually initialize the weights of the layers using the provided weight_init function
        nn.init.kaiming_uniform_(self.layer.weight)
        nn.init.kaiming_uniform_(self.output.weight)

    def forward(self, x):
        # maxnorm weight before actual forward pass
        with torch.no_grad():
            norm = self.layer.weight.norm(2, dim=0, keepdim=True) \
            .clamp(min=self.weight_constraint / 2)
            desired = torch.clamp(norm, max=self.weight_constraint)
            self.layer.weight *= (desired / norm)
        # Actual forward pass through the network
        x = self.act(self.layer(x))
        x = self.prob(self.output(x))
        return x


if __name__ == '__main__':
    df = pd.read_csv("./datasets/Pima_Indians_Diabetes/diabetes.csv")
    X = df.iloc[:, :8]
    y = df.iloc[:, -1]
    print(y.shape)
    print(y.ndim)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Reshape y to be a column vector, (n, ) -> (n, 1)
                                                           # This is necessary because BCEWithLogitsLoss expects
                                                           # the target to have the same shape as the output of
                                                           # the model, which is (n, 1) for binary classification.
                                                           # You could also use y.reshape(-1, 1) or y.view(-1, 1).
    print(y.shape)
    print(y.ndim)

    model = NeuralNetClassifier(
        module=PimaClassifier,
        criterion=nn.BCELoss,
        optimizer=optim.Adamax,
        max_epochs=100,
        batch_size=10,
        verbose=False
    )

    param_grid = {
        'module__n_neurons': [1, 5, 10, 15, 20, 25, 30],
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy',
                        verbose=1, n_jobs=-1, cv=3) # n_jobs=-1 means using
                                                    # all available cores for
                                                    # parallel processing
    grid_result = grid.fit(X, y)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, std, params in zip(means, stds, params):
        print(f"Mean: {mean}, Std: {std} using {params}")
    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
