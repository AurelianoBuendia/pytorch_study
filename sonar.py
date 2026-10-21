
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve
import torch
import torch.nn as nn

from model_utils import model_train


class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(60, 180)
        self.relu = nn.ReLU()
        self.output = nn.Linear(180, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(60, 60)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(60, 60)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(60, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.layer3(x)
        x = self.act3(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    df = pd.read_csv("./datasets/sonar.csv")
    X = df.iloc[:, :60]
    y = df.iloc[:, 60]
    print(X.head())
    print(f"y type:\n{type(y)}")
    print(f"y.head():\n{y.head()}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    print(f"type(y): {type(y)}")
    print(y.shape)
    print(y[:5])
    print(encoder.classes_)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)  # Reshape y to be a column vector
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    model_w = Wide()
    model_d = Deep()
    model_w_parameters = [p for p in model_w.parameters()]
    model_d_parameters = [p for p in model_d.parameters()]

    # Check that both models have approximately the same number of parameters
    # print(f"model_w parameters' shapes:\n{[p.shape for p in model_w_parameters]}")
    # print(f"model_d parameters' shapes:\n{[p.shape for p in model_d_parameters]}")
    # print(f"model_w parameters' shapes RESHAPED:\n{[p.reshape(-1).shape for p in model_w_parameters]}")
    # print(f"model_d parameters' shapes RESHAPED:\n{[p.reshape(-1).shape for p in model_d_parameters]}")
    print(f"model_w total number of parameters: {sum([p.reshape(-1).shape[0] for p in model_w_parameters])}")
    print(f"model_d total number of parameters: {sum([p.reshape(-1).shape[0] for p in model_d_parameters])}")

    #train-test split: Hold out the test set for final model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
    #define 5-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    cv_scores_wide = []
    n_epochs = 300
    batch_size = 10
    for train, test in kfold.split(X_train, y_train):
        #create model, train, and get accuracy
        model = Wide()
        acc = model_train(model, device, X_train[train], y_train[train], X_train[test], y_train[test],
                          n_epochs, batch_size, loss_fn=nn.BCELoss(),
                          optimizer=torch.optim.Adam(model.parameters()))
        print("Accuracy (wide): %.2f"% acc)
        cv_scores_wide.append(acc)
    cv_scores_deep = []
    for train, test in kfold.split(X_train, y_train):
        #create model, train, and get accuracy
        model = Deep()
        acc = model_train(model, device, X_train[train], y_train[train], X_train[test], y_train[test],
                          n_epochs, batch_size, loss_fn=nn.BCELoss(),
                          optimizer=torch.optim.Adam(model.parameters()))
        print("Accuracy (deep): %.2f"% acc)
        cv_scores_deep.append(acc)
    #evaluate the model
    wide_acc = np.mean(cv_scores_wide)
    wide_std = np.std(cv_scores_wide)
    deep_acc = np.mean(cv_scores_deep)
    deep_std = np.std(cv_scores_deep)
    print(f"Wide: {(float(wide_acc * 100)):.2f}% (+/- {float(wide_std * 100):.2f}%)")
    print(f"Deep: {(float(deep_acc * 100)):.2f}% (+/- {float(deep_std * 100):.2f}%)")
    #rebuild model with full set of training data
    if wide_acc > deep_acc:
        print("Retrain a wide model")
        model = Wide()
    else:
        print("Retrain a deep model")
        model = Deep()
    acc = model_train(model, device, X_train, y_train, X_test, y_test,
                      n_epochs, batch_size, loss_fn=nn.BCELoss(),
                      optimizer=torch.optim.Adam(model.parameters()))
    print(f"Final model accuracy: {acc*100:.2f}%")
    model.eval()
    with torch.no_grad():
        #Test out inference with 5 samples
        for i in range(5):
            y_pred = model(X_test[i:i+1])
            print(f"{X_test[i].numpy()} -> {y_pred[0].numpy()} " +
                  f"(expected {y_test[i].numpy()})")
    #Plot the ROC curve
    y_pred = model(X_test)
    y_test = y_test.detach().numpy()
    y_pred = y_pred.detach().numpy()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr) #ROC curve = TPR vs FPR
    plt.title("Receiver Operating Characteristics")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
