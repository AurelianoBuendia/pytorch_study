
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import im_convert


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizes image to range [-1, 1],
                                                                     # using x - mu / sigma, where mu = 0.5
                                                                     # and sigma = 0.5
])

training_dataset = datasets.CIFAR10(root='./datasets', train=True, download=False, transform=transform_train)
validation_dataset = datasets.CIFAR10(root='./datasets', train=False, download=False, transform=transform)

def show_images(data_loader, num_images=20):
    # Display a grid of images from the data loader
    # The function will fetch a batch of images and labels from the data loader,
    # convert the images to a format suitable for display, and then plot them in a grid.
    # The number of images to display can be specified with the num_images parameter.
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    fig = plt.figure(figsize=(36, 12))
    for idx in range(num_images):
        ax = fig.add_subplot(2, num_images // 2, idx + 1, xticks=[], yticks=[])
        img = im_convert(images[idx])
        ax.imshow(img)
        ax.set_title(f"{labels[idx].item()}")
    plt.show()

class LeNet(nn.Module):
    def __init__(self, i_channels=3, n_classes=10):
        super().__init__()
        self.i_channels = i_channels
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=i_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=4 * 4 * 64, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training) # Apply dropout during training to prevent overfitting
        x = self.fc2(x) # The final layer outputs raw scores (logits) for each class.
                        # These logits can be passed through a softmax function to obtain probabilities
                        # for each class if needed.
        return x

def train(model, device, optimizer, criterion, train_loader, validation_loader, n_epochs=20):
    model.train()
    running_loss_history = []
    running_corrects_history = []
    val_running_loss_history = []
    val_running_corrects_history = []
    for epoch in range(n_epochs):
        running_loss = 0.0
        running_corrects = 0.0
        val_running_loss = 0.0
        val_running_corrects = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)
        with torch.no_grad():
            for val_inputs, val_labels in validation_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                _, val_preds = torch.max(val_outputs, 1)
                val_running_loss += val_loss.item()
                val_running_corrects += torch.sum(val_preds == val_labels.data)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_corrects / len(train_loader)
        running_loss_history.append(epoch_loss)
        running_corrects_history.append(epoch_acc)

        val_epoch_loss = val_running_loss / len(validation_loader)
        val_epoch_acc = val_running_corrects / len(validation_loader)
        val_running_loss_history.append(val_epoch_loss)
        val_running_corrects_history.append(val_epoch_acc)

        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        print(f"Validation Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

    plt.plot(running_loss_history, label='Training Loss')
    plt.plot(val_running_loss_history, label='Validation Loss')
    plt.legend()
    plt.show()
    plt.plot(running_corrects_history, label='Training Accuracy')
    plt.plot(val_running_corrects_history, label='Validation Accuracy')
    plt.legend()
    plt.show()

def test(model, device, test_loader, classes, n_images=20):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    fig = plt.figure(figsize=(36, 12))
    for idx in range(n_images):
        ax = fig.add_subplot(2, n_images // 2, idx + 1, xticks=[], yticks=[])
        img = im_convert(images[idx])
        ax.imshow(img)
        ax.set_title(f"{classes[preds[idx].item()]} ({classes[labels[idx].item()]})", color=('green' if preds[idx] == labels[idx] else 'red'))
    plt.show()


if __name__ == "__main__":
    training_loader = DataLoader(training_dataset, batch_size=100, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=100, shuffle=False)
    show_images(training_loader, num_images=20)
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 20
    train(model, device, optimizer, criterion, training_loader, validation_loader, n_epochs)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    test(model, device, validation_loader, classes, n_images=20)

