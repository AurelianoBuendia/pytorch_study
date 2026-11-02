
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(8192, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        #input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        #input 32x32x32, output 32x16x16
        x = self.pool2(x)
        #input 32x16x16, output 8192
        x = self.flat(x)
        #input 8192, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        #input 512, output 10
        x = self.fc4(x)
        return x


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_for_model = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR10(root='./datasets', train=True, download=False, transform=transform_for_model)
    testset = datasets.CIFAR10(root='./datasets', train=False, download=False, transform=transform_for_model)
    batch_size = 32
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    # model = CIFAR10Model()
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # n_epochs = 20
    # for epoch in range(n_epochs):
    #     model.train()
    #     for inputs, labels in trainloader:
    #         y_pred = model(inputs)
    #         loss = loss_fn(y_pred, labels)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     acc = 0
    #     count = 0
    #     model.eval()
    #     for inputs, labels in testloader:
    #         with torch.no_grad():
    #             y_pred = model(inputs)
    #             acc += (torch.argmax(y_pred, 1) == labels).float().sum()
    #             count += len(labels)
    #     acc /= count
    #     print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}, Accuracy: {acc.item() * 100:.2f}%")
    # torch.save(model.state_dict(), "./output/cifar10_model.pth")

    model = CIFAR10Model()
    model.load_state_dict(torch.load("./output/cifar10_model.pth"))

    img = trainset.data[7]
    print(type(img))
    # Image has shape (H, W, C) -> (32, 32, 3) and is of type numpy.ndarray
    print(img.shape)
    plt.imshow(img)
    plt.show()

    X = torch.tensor([trainset.data[7]], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    print(X.shape)

    model.eval()
    with torch.no_grad():
        feature_maps = model.conv1(X)
    fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16, 8))
    # feature_maps has shape (1, 32, 32, 32) -> (batch_size, channels, height, width), so we can visualize the
    # first 32 feature maps by accessing feature_maps[0][i] for i in range(32)
    for i in range(32):
        row, col = i // 8, i % 8
        ax[row, col].imshow(feature_maps[0][i])
    plt.show()

    with torch.no_grad():
        feature_maps = model.act1(model.conv1(X))
        feature_maps = model.drop1(feature_maps)
        feature_maps = model.conv2(feature_maps)
    fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16, 8))
    # feature_maps has shape (1, 32, 32, 32) -> (batch_size, channels, height, width), so we can visualize the
    # first 32 feature maps by accessing feature_maps[0][i] for i in range(32)
    for i in range(32):
        row, col = i // 8, i % 8
        ax[row][col].imshow(feature_maps[0][i])
    plt.show()

