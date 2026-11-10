
from torch_snippets import *
import selectivesearch
from skimage.segmentation import felzenszwalb
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def extract_candidates(img):
    img_lbl, regions = selectivesearch.selective_search(img,
                                                        scale=200,
                                                        min_size=100
                                                        )
    img_area =np.prod(img.shape[:2])
    candidates = []
    for region in regions:
        if region['rect'] in candidates:
            continue
        if region['size'] < (0.05 * img_area):
            continue
        if region['size'] > (1.0 * img_area):
            continue
        x, y, w, h = region['rect']
        candidates.append(list(region['rect']))
    return candidates
def get_iou(boxA, boxB, epsilon=1e-5):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    width = (x2 - x1)
    height = (y2 - y1)
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    area_combined = area_a + area_b - area_overlap
    iou = area_overlap / (area_combined + epsilon)
    return iou


if __name__ == "__main__":
    img = Image.open("images/artur.jpg")
    print(type(img))
    img = np.array(img.convert("RGB"))
    print(type(img), img.shape)
    plt.imshow(img)
    plt.show()

    segments_fz = felzenszwalb(img, scale=200)

    subplots([img, segments_fz],
             titles = ["Original Image", "Image after \nFelzenszwalb segmentation"],
             sz=10,
             nc=2
             )

    candidates = extract_candidates(img)
    show(img, bbs=candidates)
