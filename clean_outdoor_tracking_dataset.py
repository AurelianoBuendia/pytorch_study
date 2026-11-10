
import numpy as np
import matplotlib.pyplot as plt
import json
import os


if __name__ == "__main__":
    data_folder = "./datasets/OutdoorTracking.v4i.coco"
    train_file = os.path.join(data_folder, "train/annotations.coco.json")
    annotations = {}
    try:
        with open(train_file) as f:
                annotations = json.load(f)
    except Exception as e:
        print(f"Error reading the annotations file: {e}")
        exit(1)
    print(annotations['info'])
    print(annotations.keys())
    print(len(annotations['images']))
    print(len(annotations['annotations']))
    filenames = []
    for image in annotations['images']:
        name = image['file_name']
        parts = name.split('.')
        # print(parts)
        final_name = parts[0].split('_')[0] + '_' + parts[0].split('_')[1] + '.' + parts[-1]
        filenames.append(name)
    filenames.sort()
    for name in filenames:
        print(name)
