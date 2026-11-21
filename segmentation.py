"""

1. Shape and Channel Verification for U-Net Segmentation Model with ResNet50 Backbone

Assuming an input tensor of shape (B, 3, H, W) where H and W are divisible by 32:

Stage	Operation	Output Shape	Status
Encoder	block1 (conv1 + bn1 + relu)	(B, 64, H/2, W/2)	Matches up_conv9 skip
	pool1 (maxpool)	(B, 64, H/4, W/4)	Valid
	block2 (layer1)	(B, 256, H/4, W/4)	Matches up_conv8 skip
	block3 (layer2)	(B, 512, H/8, W/8)	Matches up_conv7 skip
	block4 (layer3)	(B, 1024, H/16, W/16)	Matches up_conv6 skip
	block5 (layer4)	(B, 2048, H/32, W/32)	Valid
Bottleneck	conv_bottleneck	(B, 2048, H/32, W/32)	Valid
Decoder	up_conv6 + cat(block4) → conv6	(B, 1024, H/16, W/16)	1024 + 1024 = 2048 in
	up_conv7 + cat(block3) → conv7	(B, 512, H/8, W/8)	512 + 512 = 1024 in
	up_conv8 + cat(block2) → conv8	(B, 256, H/4, W/4)	256 + 256 = 512 in
	up_conv9 + cat(block1) → conv9	(B, 64, H/2, W/2)	64 + 64 = 128 in
	up_conv10 → conv10	(B, 32, H, W)	Valid
Output	conv11 (1x1 Conv)	(B, 12, H, W)	Clean raw logits

To freeze and unfreeze the ResNet50 encoder cleanly, add a helper method to your UNet_ResNet50 class that
toggles requires_grad across all encoder blocks and keeps BatchNorm layers in .eval() mode when frozen.
1. Add Freeze/Unfreeze Methods to UNet_ResNet50
Because your model references the ResNet50 layers via self.block1 through self.block5 (and self.encoder), the
safest approach is to group the encoder blocks into a list and loop over them.

2. Why Overriding train() Matters for ResNet50
In PyTorch, setting param.requires_grad = False stops gradient updates for weights and biases, but nn.BatchNorm2d
layers will still update their running mean and variance every time model.train() is called unless they are
explicitly kept in .eval() mode.
Overriding train() as shown above guarantees that calling model.train() at the start of every training epoch does
not accidentally corrupt the pretrained ImageNet BatchNorm statistics while your encoder is frozen.


3. Two-Stage Transfer Learning Workflow
A standard transfer learning schedule warms up the randomly initialized decoder first before fine-tuning
 the full network:
Stage 1: Train decoder only (e.g., first 5–10 epochs)
Stage 2: Unfreeze encoder and fine-tune end-to-end
When unfreezing, re-create the optimizer (or add a new parameter group) with a 10× smaller learning rate so
large gradients don't overwrite the pretrained ResNet50 features.

"""

import os
# if not os.path.exists('dataset1'):
#    !wget -q wget -q -O dataset1.zip "https://www.dropbox.com/scl/fi/jv072cyo3y62f6ykw54l8/dataset1.zip?rlkey=mf1tmtrnd6ys7m7oi4ygtcro9&st=vj8bwlik&dl=1"
#    !unzip -q dataset1.zip
#    !rm dataset1.zip
#    %pip install -q torch_snippets pytorch_model_summary lovely_tensors

from torchvision import transforms
from sklearn.model_selection import train_test_split
import cv2
from torch_snippets import *
import torch
import torch.nn as nn
import torchvision
from torchvision.models import vgg16_bn, resnet50, ResNet50_Weights

init_torch()

tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
])

class SegData(Dataset):
    def __init__(self, split, data_path):
        self.images_path = data_path + "/images_prepped_"
        self.annotations_path = data_path + "/annotations_prepped_"
        self.items = stems(f'{self.images_path}{split}')
        self.split = split

    def __len__(self):
        return len(self.items)

    def __getitem__(self, ix):
        image = read(f'{self.images_path}{self.split}/{self.items[ix]}.png', 1)
        image = cv2.resize(image, (224, 224))
        mask = read(f'{self.annotations_path}{self.split}/{self.items[ix]}.png', 0)
        mask = cv2.resize(mask, (224, 224))
        return image, mask

    def choose(self):
        return self[randint(len(self))]

    def collate_fn(self, batch):
        # batch is a list of tuples (image, mask)
        # zip(*batch) will unzip the list of tuples into two lists: one for images and one for masks
        # list(zip(*batch)) will convert the zip object into a list of tuples
        ims, masks = list(zip(*batch))
        # [tfms(im.copy() / 255.)[None] for im in ims] will apply the transformations to each image and add
        # a new dimension at the beginning
        # torch.cat([...]) will concatenate the list of tensors along the first dimension (batch dimension)
        ims = torch.cat([tfms(im.copy() / 255.)[None] for im in ims]).float().to(device)
        ce_masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long().to(device)
        return ims, ce_masks

class UNet(nn.Module):
    def __init__(self, pretrained=True, out_channels=12):
        super().__init__()

        self.encoder = vgg16_bn(weights=torchvision.models.VGG16_BN_Weights.DEFAULT).features
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])

        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.conv_bottleneck = conv(512, 1024)

        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = conv(512 + 512, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = conv(256 + 512, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = conv(128 + 256, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = conv(64 + 128, 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = conv(32 + 64, 32)
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)

        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)

        x = self.up_conv6(x)
        x = torch.cat([x, block5], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)

        x = self.conv11(x)

        return x

class UNet_ResNet50(nn.Module):
    """
    To convert the UNet implementation to use a ResNet50 backbone, you need to replace the sequential VGG feature
    slices with ResNet50's named stages and update the decoder's input channel counts to match ResNet50's bottleneck
    expansions (64, 256, 512, 1024, and 2048).
    Key Structural Changes
    • Feature Extraction: Unlike VGG16, ResNet50 does not have a single .features list. You must extract its initial
    conv1/bn1/relu stem and its four residual stages (layer1 through layer4).
    • Spatial Resolution: ResNet50's initial conv1 uses stride=2, immediately downsampling the input to H/2 × W/2
    before maxpool reduces it to H/4 × W/4. Because there is no full-resolution (H × W) feature map in ResNet50's
    encoder, your final decoder block (up_conv10) upsamples back to the original input size (H × W) without a skip
    connection (or concatenates with the raw input x if desired).
    • Channel Dimensions: ResNet50 outputs significantly deeper feature maps at each stage than VGG16, requiring
    updated in_channels for your conv blocks.

    Channel & Resolution Mapping
    Stage	VGG16-BN Channels	ResNet50 Channels	Output Resolution
    block1	64 (H x W)	64 (H/2 x W/2)	H/2 (ResNet)
    block2	128 (H/2 x W/2)	256 (H/4 x W/4)	H/4 (ResNet)
    block3	256 (H/4 x W/4)	512 (H/8 x W/8)	H/8 (ResNet)
    block4	512 (H/8 x W/8)	1024 (H/16 x W/16)	H/16 (ResNet)
    bottleneck	512 (H/32 x W/32)	2048 (H/32 x W/32)	H/32 (Both)

    Note: This assumes the up_conv(in_c, out_c) helper doubles spatial dimensions (scale_factor=2 or
    ConvTranspose2d with stride=2) and your input height/width are divisible by 32.
    """
    def __init__(self, pretrained=True, out_channels=12):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        encoder = resnet50(weights=weights)

        # Encoder blocks
        self.block1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool1 = encoder.maxpool
        self.block2 = encoder.layer1
        self.block3 = encoder.layer2
        self.block4 = encoder.layer3
        self.block5 = encoder.layer4

        # Store encoder modules in a list for easy freezing/unfreezing
        self.encoder_blocks = [
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5,
        ]

        # Bottleneck
        self.conv_bottleneck = conv(2048, 2048)

        # Decoder blocks (matching skip connection channels)
        self.up_conv6 = up_conv(2048, 1024)
        self.conv6 = conv(1024 + 1024, 1024)  # Concat with block4 (1024)

        self.up_conv7 = up_conv(1024, 512)
        self.conv7 = conv(512 + 512, 512)  # Concat with block3 (512)

        self.up_conv8 = up_conv(512, 256)
        self.conv8 = conv(256 + 256, 256)  # Concat with block2 (256)

        self.up_conv9 = up_conv(256, 64)
        self.conv9 = conv(64 + 64, 64)  # Concat with block1 (64)

        # Final upsample to restore H/2 -> H
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = conv(32, 32)

        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)

    def set_encoder_trainable(self, trainable: bool):
        """Freezes or unfreezes the ResNet50 encoder weights."""
        for block in self.encoder_blocks:
            for param in block.parameters():
                param.requires_grad = trainable
            # Keep BatchNorm running stats frozen when encoder is frozen
            if not trainable:
                block.eval()
            else:
                block.train()

    def train(self, mode: bool = True):
        """Overrides model.train() so frozen BatchNorm layers stay in eval mode."""
        super().train(mode)
        if mode:
            for block in self.encoder_blocks:
                # Check if the block is frozen
                if not any(p.requires_grad for p in block.parameters()):
                    block.eval()
        return self

    def forward(self, x):
        # Encoder
        block1 = self.block1(x)  # H/2, 64 ch
        p1 = self.pool1(block1)  # H/4, 64 ch
        block2 = self.block2(p1)  # H/4, 256 ch
        block3 = self.block3(block2)  # H/8, 512 ch
        block4 = self.block4(block3)  # H/16, 1024 ch
        block5 = self.block5(block4)  # H/32, 2048 ch

        # Bottleneck
        x = self.conv_bottleneck(block5)  # H/32, 2048 ch

        # Decoder
        x = self.up_conv6(x)  # H/16, 1024 ch
        x = torch.cat([x, block4], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)  # H/8, 512 ch
        x = torch.cat([x, block3], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)  # H/4, 256 ch
        x = torch.cat([x, block2], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)  # H/2, 64 ch
        x = torch.cat([x, block1], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)  # H, 32 ch
        x = self.conv10(x)

        x = self.conv11(x)  # H, out_channels
        return x

def conv(in_channels, out_channels):
    return nn.Sequential(
        # Because nn.Conv2d is immediately followed by nn.BatchNorm2d, the convolutional bias term gets canceled out
        # by batch normalization's mean subtraction. Setting bias=False in nn.Conv2d saves a small amount of memory
        # and computation without changing the math
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def up_conv(in_channels, out_channels):
    return nn.Sequential(
        # Because nn.Conv2d is immediately followed by nn.BatchNorm2d, the convolutional bias term gets canceled out
        # by batch normalization's mean subtraction. Setting bias=False in nn.Conv2d saves a small amount of memory
        # and computation without changing the math
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def UnetLoss(preds, targets):
    loss_fn = nn.CrossEntropyLoss()
    ce_loss = loss_fn(preds, targets)
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    return ce_loss, acc

def train_batch(model, data, optimizer, criterion):
    model.train()
    ims, ce_masks = data
    _masks = model(ims)
    optimizer.zero_grad()
    loss, acc = criterion(_masks, ce_masks)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()

@torch.no_grad()
def validate_batch(model, data, criterion):
    model.eval()
    ims, masks = data
    _masks = model(ims)
    loss, acc = criterion(_masks, masks)
    return loss.item(), acc.item()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(2, 2), stride=2, padding=0)
    input = torch.ones(1, 1, 3, 3)
    output = m(input)
    # In PyTorch, the output spatial dimension for nn.ConvTranspose2d is calculated as:
    # O = (I - 1) * S - 2P + D * (K - 1) + P_out + 1
    # E.g., for the given parameters:
    # I = 3 (input size), S = 2 (stride), P = 0 (padding), P_out = 0 (output padding), D = 1 (dilation), K = 2 (kernel size)
    # O = (3 - 1) * 2 - 2 * 0 + 1 * (2 - 1) + 0 + 1 = 2 * 2 - 0 + 1 + 0 + 1 = 4 + 1 + 1 = 6
    # batch size = 1 and output channels = 1 carry over directly to the final shape: torch.Size([1, 1, 6, 6]).
    print(output.shape)

    data_path = "datasets/streets_segmentation"

    trn_ds = SegData('train', data_path)
    val_ds = SegData('test', data_path)
    trn_dl = DataLoader(trn_ds, batch_size=4, shuffle=True, collate_fn=trn_ds.collate_fn)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=val_ds.collate_fn)

    show(trn_ds[10][0])

    n_epochs = 10
    criterion = UnetLoss

    # Training loop for UNet with ResNet50 backbone
    # model = UNet_ResNet50().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    #
    # log = Report(n_epochs)
    # for ex in range(n_epochs):
    #     N = len(trn_dl)
    #     for bx, data in enumerate(trn_dl):
    #         loss, acc = train_batch(model, data, optimizer, criterion)
    #         log.record(ex + (bx + 1) / N, trn_loss=loss, trn_acc=acc, end='\r')
    #     N = len(val_dl)
    #     for bx, data in enumerate(val_dl):
    #         loss, acc = validate_batch(model, data, criterion)
    #         log.record(ex + (bx + 1) / N, val_loss=loss, val_acc=acc, end='\r')
    #     log.report_avgs(ex + 1)
    #
    # log.plot_epochs(['trn_loss', 'val_loss'])
    #
    # im, mask = next(iter(val_dl))
    # _mask = model(im)
    # _, _mask = torch.max(_mask, dim=1)
    # subplots([im[0].permute(1, 2, 0).detach().cpu()[:, :, 0], mask.permute(1, 2, 0).detach().cpu()[:, :, 0],
    #           _mask.permute(1, 2, 0).detach().cpu()[:, :, 0]],
    #          nc=3, titles=['Original image', 'Original mask', 'Predicted mask'])

    # Two-Stage Transfer Learning Workflow
    # Stage 1: Train decoder only (e.g., first 5–10 epochs)
    model = UNet_ResNet50(pretrained=True, out_channels=12).to(device)
    model.set_encoder_trainable(False)
    # Pass only trainable parameters to the optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    # Stage 2: Unfreeze encoder and fine-tune end-to-end
    model.set_encoder_trainable(True)
    # Use differential learning rates: lower for encoder, higher for decoder
    encoder_params = [p for b in model.encoder_blocks for p in b.parameters()]
    encoder_param_ids = set(map(id, encoder_params))
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": 1e-4},
        {"params": decoder_params, "lr": 1e-3},
    ])

    log = Report(n_epochs)
    for ex in range(n_epochs):
        N = len(trn_dl)
        for bx, data in enumerate(trn_dl):
            loss, acc = train_batch(model, data, optimizer, criterion)
            log.record(ex + (bx + 1) / N, trn_loss=loss, trn_acc=acc, end='\r')
        N = len(val_dl)
        for bx, data in enumerate(val_dl):
            loss, acc = validate_batch(model, data, criterion)
            log.record(ex + (bx + 1) / N, val_loss=loss, val_acc=acc, end='\r')
        log.report_avgs(ex + 1)

    log.plot_epochs(['trn_loss', 'val_loss'])

    im, mask = next(iter(val_dl))
    _mask = model(im)
    _, _mask = torch.max(_mask, dim=1)
    subplots([im[0].permute(1, 2, 0).detach().cpu()[:, :, 0], mask.permute(1, 2, 0).detach().cpu()[:, :, 0],
              _mask.permute(1, 2, 0).detach().cpu()[:, :, 0]],
             nc=3, titles=['Original image', 'Original mask', 'Predicted mask'])
