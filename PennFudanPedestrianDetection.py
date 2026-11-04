
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from PIL import Image
import os


def get_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    return image

class PennFudanDataset(Dataset):
    def __init__(self, root_dir, transform=torchvision.transforms.ToTensor()):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in sorted(os.listdir(os.path.join(root_dir, 'PNGImages'))) if f.endswith('.png')]
        self.mask_files = [f for f in sorted(os.listdir(os.path.join(root_dir, 'PedMasks'))) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, 'PNGImages', self.image_files[index])
        mask_path = os.path.join(self.root_dir, 'PedMasks', self.mask_files[index])

        image = get_image(img_path)
        mask = Image.open(mask_path)
        mask = np.array(mask)

        # Get distinct object IDs from the mask (skip the background which is 0)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]  # remove background
        num_objs = len(obj_ids)
        masks = np.zeros((num_objs, mask.shape[0], mask.shape[1])) # Create a binary mask for each object
        for i in range(num_objs):
            masks[i][mask == i + 1] = True  # Set the mask for the current object

        # Get bounding boxes for each object
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # Pedestrians class is equal to 1
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        is_crowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "area": area,
            "is_crowd": is_crowd
        }

        if self.transform:
            image = self.transform(image)

        return image, target

def get_instance_segmentation_model(num_classes):
    # Load a pre-trained Mask R-CNN model on COCO dataset
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # 1. Get the number of input features for the box classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained box predictor head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 2. Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the pre-trained mask predictor head
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    return model

# ==========================================
# 4. RUN AN INSTANCE PASS
# ==========================================
# Collate function to handle variable sized boxes/masks in a batch
def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    DATA_DIR = "./datasets/PennFudanPed"
    # Initialize dataset and loader
    dataset = PennFudanDataset(DATA_DIR)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 2 classes: Background (0) and Pedestrian (1)
    model = get_instance_segmentation_model(num_classes=2)

    # Set model to training mode to fetch loss metrics
    model.train()

    # Fetch one batch of data
    images, targets = next(iter(data_loader))

    # Forward pass through the model
    loss_dict = model(images, targets)

    # Display the output metrics
    print("\n--- Training Step Success ---")
    print("Loss breakdown for the current batch:")
    for loss_name, loss_value in loss_dict.items():
        print(f" * {loss_name}: {loss_value.item():.4f}")
