
import torch
import torch.optim as optim
from torchvision import models
import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import load_image, im_convert_squeeze


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_features(image, model):
    layers = {'0': 'conv1_1',
              '5': 'conv2_1',
              '10': 'conv3_1',
              '19': 'conv4_1',
              '21': 'conv4_2',  # Content extraction layer. The others are used for style extraction.
              '28': 'conv5_1'}
    features = {}
    for name, layer in model._modules.items():
        image = layer(image)
        if name in layers:
            features[layers[name]] = image
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


if __name__ == "__main__":
    vgg = models.vgg19(pretrained=True).features
    vgg.to(device)
    print(vgg)
    for param in vgg.parameters():
        param.requires_grad_(False)
    vgg.to(device)
    content = load_image('./images/City.jpg').to(device)
    style = load_image('./images/StarryNight.jpg').to(device)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    conv_content = im_convert_squeeze(content)
    ax1.imshow(conv_content)
    ax1.axis("off")
    conv_style = im_convert_squeeze(style)
    ax2.imshow(conv_style)
    ax2.axis("off")
    plt.show()

    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    style_weights = {'conv1_1': 1.,
                     'conv2_1': 0.75,
                     'conv3_1': 0.2,
                     'conv4_1': 0.2,
                     'conv5_1': 0.2}

    content_weight = 1  # alpha
    style_weight = 1e6  # beta
    target = content.clone().requires_grad_(True).to(device)
    show_every = 300
    optimizer = optim.Adam([target], lr=0.003)
    steps = 2100

    height, width, channels = im_convert_squeeze(target).shape
    image_array = np.empty(shape=(300, height, width, channels))
    capture_frame = steps / 300
    counter = 0
    for ii in range(1, steps + 1):
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        style_loss = 0

        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            _, d, h, w = target_feature.shape
            style_loss += layer_style_loss / (d * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if ii % show_every == 0:
            print('Total loss: ', total_loss.item())
            print('Iteration: ', ii)
            plt.imshow(im_convert_squeeze(target))
            plt.axis("off")
            plt.show()

        if ii % capture_frame == 0:
            image_array[counter] = im_convert_squeeze(target)
            counter = counter + 1

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(im_convert_squeeze(content))
    ax1.axis('off')
    ax2.imshow(im_convert_squeeze(style))
    ax2.axis('off')
    ax3.imshow(im_convert_squeeze(target))
    ax3.axis('off')
    plt.show()

    frame_height, frame_width, _ = im_convert_squeeze(target).shape
    vid = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

    for i in range(0, 300):
        img = image_array[i]
        img = img * 255
        img = np.array(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        vid.write(img)

    vid.release()
