"""Convert COCO detection annotations into a YOLO-style dataset."""

import argparse
import json
import math
import shutil
import tempfile
import warnings
from pathlib import Path


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _read_split(annotation_file, image_root):
    annotation_file = Path(annotation_file).resolve()
    image_root = Path(image_root).resolve()
    with annotation_file.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{annotation_file}: expected a COCO JSON object")
    for key in ("images", "annotations", "categories"):
        if not isinstance(data.get(key), list):
            raise ValueError(f"{annotation_file}: {key} must be a list")
    return data, image_root


def _integer(value, description):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{description} must be an integer")
    return value


def _categories(data, annotation_file):
    result = {}
    for category in data["categories"]:
        category_id = _integer(category["id"], "category id")
        name = category["name"]
        if category_id in result:
            raise ValueError(f"{annotation_file}: duplicate category id {category_id}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{annotation_file}: category name must be nonempty")
        result[category_id] = name
    if not result:
        raise ValueError(f"{annotation_file}: no categories")
    return result


def _positive_number(value, description):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{description} must be finite and positive")
    return value


def _box(annotation, width, height):
    box = annotation.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("bbox must contain [x, y, width, height]")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        raise ValueError("bbox coordinates must be finite numbers")
    x, y, w, h = box
    _positive_number(w, "bbox width")
    _positive_number(h, "bbox height")
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox does not overlap its image")
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height)


def _convert_split(data, image_root, output, split, category_indices):
    images = {}
    for image in data["images"]:
        image_id = _integer(image["id"], "image id")
        if image_id in images:
            raise ValueError(f"{split}: duplicate image id {image_id}")
        width = _positive_number(image["width"], "image width")
        height = _positive_number(image["height"], "image height")
        file_name = image["file_name"]
        if not isinstance(file_name, str) or not file_name:
            raise ValueError(f"{split}: image file_name must be nonempty")
        source = (image_root / file_name).resolve()
        if not source.is_relative_to(image_root) or not source.is_file():
            raise ValueError(f"{split}: image is missing or outside its image root: {file_name}")
        if source.suffix.lower() not in _IMAGE_SUFFIXES:
            raise ValueError(f"{split}: unsupported image extension: {file_name}")
        images[image_id] = (source, width, height)
    if not images:
        raise ValueError(f"{split}: no images")

    labels = {image_id: [] for image_id in images}
    crowd_count = 0
    for annotation in data["annotations"]:
        image_id = _integer(annotation["image_id"], "annotation image_id")
        category_id = _integer(annotation["category_id"], "annotation category_id")
        if image_id not in images:
            raise ValueError(f"{split}: annotation references unknown image id {image_id}")
        if category_id not in category_indices:
            raise ValueError(f"{split}: annotation references unknown category id {category_id}")
        crowd = annotation.get("iscrowd", 0)
        if crowd not in (0, 1):
            raise ValueError(f"{split}: iscrowd must be 0 or 1")
        if crowd:
            crowd_count += 1
            continue
        _, width, height = images[image_id]
        try:
            coords = _box(annotation, width, height)
        except ValueError as exc:
            raise ValueError(f"{split}: invalid annotation for image {image_id}: {exc}") from exc
        labels[image_id].append(
            f"{category_indices[category_id]} " + " ".join(f"{value:.8f}" for value in coords)
        )

    if crowd_count:
        warnings.warn(f"{split}: omitted {crowd_count} crowd annotations", stacklevel=2)

    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for image_id, (source, _, _) in images.items():
        shutil.copyfile(source, image_dir / f"{image_id}{source.suffix.lower()}")
        lines = labels[image_id]
        (label_dir / f"{image_id}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )


def convert_coco(train_annotations, train_images, val_annotations, val_images, output):
    """Create a new dataset directory from COCO train and validation splits.

    Category IDs are mapped to consecutive indices in ascending ID order. Existing
    output directories are never overwritten.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    train, train_root = _read_split(train_annotations, train_images)
    val, val_root = _read_split(val_annotations, val_images)
    categories = _categories(train, train_annotations)
    if _categories(val, val_annotations) != categories:
        raise ValueError("train and validation category IDs and names must match")
    category_indices = {category_id: index for index, category_id in enumerate(sorted(categories))}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".coco-conversion-", dir=output.parent) as tmp:
        staging = Path(tmp) / "dataset"
        staging.mkdir()
        _convert_split(train, train_root, staging, "train", category_indices)
        _convert_split(val, val_root, staging, "val", category_indices)
        config = {
            "path": str(output),
            "train": "images/train",
            "val": "images/val",
            "names": [categories[category_id] for category_id in sorted(categories)],
        }
        # JSON is a subset of YAML and preserves arbitrary class names safely.
        (staging / "data.yaml").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (staging / "category_ids.json").write_text(
            json.dumps({str(key): value for key, value in category_indices.items()}, indent=2) + "\n",
            encoding="utf-8",
        )
        # Reserve the destination without replacing a concurrently created directory.
        output.mkdir()
        for item in staging.iterdir():
            item.rename(output / item.name)
    return output / "data.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-annotations", required=True, type=Path)
    parser.add_argument("--train-images", required=True, type=Path)
    parser.add_argument("--val-annotations", required=True, type=Path)
    parser.add_argument("--val-images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(convert_coco(args.train_annotations, args.train_images,
                       args.val_annotations, args.val_images, args.output))


if __name__ == "__main__":
    main()
