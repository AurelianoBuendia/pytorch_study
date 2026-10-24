
from PIL import Image
import numpy as np
from torchvision import transforms
import sklearn.datasets as datasets
import torch


def mape_loss(output, target):
    # Mean Absolute Percentage Error (MAPE) loss function
    # output: predicted values (tensor)
    # target: true values (tensor)
    # Returns the mean absolute percentage error between the output and target tensors.
    return torch.mean(torch.abs((target - output) / target))

def make_circles_dataset(n_samples=1000, factor=0.5, noise=0.1):
    # Create circles dataset as tensors
    # Each row in X is the (x, y) coordinate of one point.  The outer circle has radius 1 (points like (cos θ, sin θ)),
    # and the inner circle has radius factor (default 0.8), so its points are (factor * cos θ, factor * sin θ).
    # The label y tells you which circle each row belongs to: 0 = outer, 1 = inner.
    X, y = datasets.make_circles(n_samples, factor=factor, noise=noise)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    return X, y

# In general, images are represented as 3D arrays (H, W, C) where H is the height, W is the width,
# and C is the number of channels (e.g., 3 for RGB images). However, PyTorch models expect input tensors to have
# the shape (N, C, H, W), where N is the batch size. Therefore, for PyTorch usage we need to convert images from
# (H, W, C) to (C, H, W) and add a batch dimension. For visualization of tensors, we need to reverse this process.
# Reference: https://pytorch.org/docs/stable/torchvision/transforms.html
# PIL Image:          (H, W)       ←  2D, grayscale
#                     (H, W, C)    ←  3D, RGB (PIL/OpenCV convention)
# After ToTensor:     (C, H, W)    ←  PyTorch convention, channels moved to front
# After unsqueeze(0): (N, C, H, W)  ←  ready for the model

def im_convert(tensor):
    # Converts a (1, C, H, W) tensor to a (H, W, C) numpy image for visualization
    # Note that the input tensor is expected to be in the range [-1, 1] due to normalization
    # The function will convert it back to the range [0, 1].
    # Also, the function will detach the tensor from the computation graph and move it to CPU if necessary,
    # since NumPy does not support GPU tensors.
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0) # Convert from (C, H, W) to (H, W, C)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5)) # Unnormalize the image from [-1, 1] to [0, 1]
    image = image.clip(0, 1) # Clip the values to be in the range [0, 1] to avoid any artifacts due to unnormalization
    return image

def im_convert_squeeze(tensor):
    # Converts a (1, C, H, W) tensor to a (H, W, C) numpy image for visualization
    # The input tensor is expected to be in the range [-1, 1] due to normalization
    # The function will convert it back to the range [0, 1].
    # Also, the function will detach the tensor from the computation graph and move it to CPU if necessary,
    # since NumPy does not support GPU tensors.
    image = tensor.cpu().clone().detach()
    image = image.numpy().squeeze(0) # Remove the batch dimension
    image = image.transpose(1, 2, 0) # Convert from (C, H, W) to (H, W, C)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5)) # Unnormalize the image from [-1, 1] to [0, 1]
    image = image.clip(0, 1) # Clip the values to be in the range [0, 1] to avoid any artifacts due to unnormalization
    return image

def load_image(img_path, max_size=400, shape=None):
    # Load an image from the specified path, resize it to a maximum size, and convert it to a tensor
    # Returns a (1, C, H, W) 4D input tensor containing the image, suitable for input to a PyTorch model.
    image = Image.open(img_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape is not None:
        size = shape
    # torchvision.transforms.Compose returns a single callable transform object that chains a list of transforms
    # together, applying each in sequence such that the output of one becomes the input for the next.
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))])
    # unsqueeze(0) adds a new dimension at index 0 to a tensor, effectively converting a single image tensor from
    # shape (C, H, W) to a batched shape (1, C, H, W). This is necessary because most pre-trained PyTorch models
    # expect a (N, C, H, W) 4D input tensor representing a batch of images rather than a single image.
    image = in_transform(image).unsqueeze(0)
    return image

