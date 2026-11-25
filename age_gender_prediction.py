
import torch
import numpy as np
import cv2
import pandas as pd
import glob
import time
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models, datasets
from torchsummary import summary


IMAGE_SIZE = 224

class GenderAgeClass(Dataset):
    def __init__(self, df, img_path, tfms=None):
        self.df = df
        self.img_path = img_path
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        f = self.df.iloc[index].squeeze()
        file = self.img_path + f.file
        gen = f.gender == 'Female'
        age = f.age
        img = cv2.imread(file)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, age, gen

    def preprocess_image(self, im):
        im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
        im = torch.tensor(im).permute(2,0,1)
        im = self.normalize(im / 255.)
        return im[None]

    def collate_fn(self, batch):
        """preprocess images, ages and genders"""
        ims, ages, genders = [], [], []
        for im, age, gender in batch:
            im = self.preprocess_image(im)
            ims.append(im)

            ages.append(float(int(age) / 80))
            genders.append(float(gender))

        ages, genders = [torch.tensor(x).to(device).float() for x in [ages, genders]]
        ims = torch.cat(ims).to(device)

        return ims, ages, genders

class AgeGenderClassifier(nn.Module):
    def __init__(self):
        super(AgeGenderClassifier, self).__init__()
        self.intermediate = nn.Sequential(
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,64),
            nn.ReLU(),
        )
        self.age_classifier = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.gender_classifier = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.intermediate(x)
        age = self.age_classifier(x)
        gender = self.gender_classifier(x)
        return gender, age

def get_model():
    full_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # Freeze parameters so we don't backprop through them
    for param in full_model.parameters():
        param.requires_grad = False
    full_model.avgpool = nn.Sequential(
        nn.Conv2d(512,512, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten()
    )
    full_model.classifier = AgeGenderClassifier()
    gender_criterion = nn.BCELoss()
    age_criterion = nn.L1Loss()
    loss_functions = gender_criterion, age_criterion
    optimizer = torch.optim.Adam(full_model.parameters(), lr=1e-4)
    return full_model, loss_functions, optimizer

def train_batch(data, model, optimizer, criteria):
    model.train()
    ims, age, gender = data
    optimizer.zero_grad()
    pred_gender, pred_age = model(ims)
    gender_criterion, age_criterion = criteria
    gender_loss = gender_criterion(pred_gender.squeeze(), gender)
    age_loss = age_criterion(pred_age.squeeze(), age)
    total_loss = gender_loss + age_loss
    total_loss.backward()
    optimizer.step()
    return total_loss

def validate_batch(data, model, criteria):
    model.eval()
    ims, age, gender = data
    with torch.no_grad():
      pred_gender, pred_age = model(ims)
    gender_criterion, age_criterion = criteria
    gender_loss = gender_criterion(pred_gender.squeeze(), gender)
    age_loss = age_criterion(pred_age.squeeze(), age)
    total_loss = gender_loss + age_loss
    pred_gender = (pred_gender > 0.5).squeeze()
    gender_acc = (pred_gender == gender).float().sum()
    age_mae = torch.abs(age - pred_age).float().sum()
    return total_loss, gender_acc, age_mae


if __name__ == '__main__':
    trn_df = pd.read_csv('datasets/FairFace/fairface-labels-train.csv')
    val_df = pd.read_csv('datasets/FairFace/fairface-labels-val.csv')
    print(trn_df.head())
    print(val_df.head())

    img_path = 'datasets/FairFace/fairface-img-margin025-trainval/'
    trn = GenderAgeClass(trn_df, img_path)
    val = GenderAgeClass(val_df, img_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader = DataLoader(trn, batch_size=32, shuffle=True, drop_last=True, collate_fn=trn.collate_fn)
    test_loader = DataLoader(val, batch_size=32, collate_fn=val.collate_fn)
    a, b, c, = next(iter(train_loader))
    print(a.shape, b.shape, c.shape)

    model, loss_functions, optimizer = get_model()
    summary(model, input_size=(3, 224, 224), device=None)

    model, criteria, optimizer = get_model()
    val_gender_accuracies = []
    val_age_maes = []
    train_losses = []
    val_losses = []

    n_epochs = 5
    best_test_loss = 1000
    start = time.time()

    for epoch in range(n_epochs):
        epoch_train_loss, epoch_test_loss = 0, 0
        val_age_mae, val_gender_acc, ctr = 0, 0, 0
        _n = len(train_loader)
        for ix, data in enumerate(train_loader):
            # if ix == 100: break
            loss = train_batch(data, model, optimizer, criteria)
            epoch_train_loss += loss.item()
            print(f"Epoch {epoch + 1}/{n_epochs} - Batch {ix + 1}/{_n} - Train Loss: {loss.item():.4f}")

        for ix, data in enumerate(test_loader):
            # if ix == 10: break
            loss, gender_acc, age_mae = validate_batch(data, model, criteria)
            epoch_test_loss += loss.item()
            val_age_mae += age_mae
            val_gender_acc += gender_acc
            ctr += len(data[0])
            print(f"Epoch {epoch + 1}/{n_epochs} - Batch {ix + 1}/{_n} - Test Loss: {loss.item():.4f}")

        val_age_mae /= ctr
        val_gender_acc /= ctr
        epoch_train_loss /= len(train_loader)
        epoch_test_loss /= len(test_loader)

        elapsed = time.time() - start
        best_test_loss = min(best_test_loss, epoch_test_loss)
        print('{}/{} ({:.2f}s - {:.2f}s remaining)'.format(epoch + 1, n_epochs, time.time() - start,
                                                           (n_epochs - epoch) * (elapsed / (epoch + 1))))
        info = f'''Epoch: {epoch + 1:03d}\tTrain Loss: {epoch_train_loss:.3f}\tTest: {epoch_test_loss:.3f}\tBest Test Loss: {best_test_loss:.4f}'''
        info += f'\nGender Accuracy: {val_gender_acc * 100:.2f}%\tAge MAE: {val_age_mae:.2f}\n'
        print(info)

        val_gender_accuracies.append(val_gender_acc)
        val_age_maes.append(val_age_mae)

    epochs = np.arange(1, len(val_gender_accuracies) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax = ax.flat
    ax[0].plot(epochs, [v.item() for v in val_gender_accuracies], 'bo')
    ax[1].plot(epochs, [v.item() for v in val_age_maes], 'r')
    ax[0].set_xlabel('Epochs')
    ax[1].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[1].set_ylabel('MAE')
    ax[0].set_title('Validation Gender Accuracy')
    ax[0].set_title('Validation Age Mean-Absolute-Error')
    plt.show()

    im = cv2.imread('datasets/FairFace/fairface-img-margin025-trainval/val/10110.jpg')
    im = trn.preprocess_image(im).to(device)
    gender, age = model(im)
    pred_gender = gender.to('cpu').detach().numpy()
    pred_age = age.to('cpu').detach().numpy()
    im = cv2.imread('datasets/FairFace/fairface-img-margin025-trainval/val/10110.jpg')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    print('predicted gender:', np.where(pred_gender[0][0] < 0.5, 'Male', 'Female'), '; Predicted age',
          int(pred_age[0][0] * 80))





