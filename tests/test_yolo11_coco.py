import json
import tempfile
import unittest
import warnings
from pathlib import Path

from yolo11.coco import convert_coco


class CocoConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "source"
        self.images.mkdir()
        (self.images / "first.jpg").write_bytes(b"first")
        (self.images / "second.png").write_bytes(b"second")
        self.data = {
            "images": [
                {"id": 10, "file_name": "first.jpg", "width": 100, "height": 50},
                {"id": 11, "file_name": "second.png", "width": 100, "height": 50},
            ],
            "categories": [{"id": 9, "name": "dog"}, {"id": 2, "name": "cat"}],
            "annotations": [
                {"image_id": 10, "category_id": 9, "bbox": [-10, 5, 40, 20]},
            ],
        }
        self.train = self.root / "train.json"
        self.val = self.root / "val.json"
        self.output = self.root / "out"
        self.write_data()

    def write_data(self):
        for path in (self.train, self.val):
            path.write_text(json.dumps(self.data), encoding="utf-8")

    def convert(self):
        return convert_coco(self.train, self.images, self.val, self.images, self.output)

    def test_conversion_clips_boxes_and_preserves_empty_images(self):
        config_path = self.convert()
        config = json.loads(config_path.read_text())
        self.assertEqual(config["names"], ["cat", "dog"])
        self.assertEqual(config["path"], str(self.output))
        self.assertEqual(json.loads((self.output / "category_ids.json").read_text()), {"2": 0, "9": 1})
        for split in ("train", "val"):
            self.assertEqual((self.output / "labels" / split / "10.txt").read_text(),
                             "1 0.15000000 0.30000000 0.30000000 0.40000000\n")
            self.assertEqual((self.output / "labels" / split / "11.txt").read_text(), "")
            self.assertEqual((self.output / "images" / split / "10.jpg").read_bytes(), b"first")

    def test_crowd_annotations_are_reported_and_omitted(self):
        self.data["annotations"].append(
            {"image_id": 10, "category_id": 2, "bbox": [1, 1, 10, 10], "iscrowd": 1}
        )
        self.write_data()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.convert()
        self.assertEqual(len(caught), 2)
        self.assertIn("omitted 1 crowd", str(caught[0].message))
        self.assertEqual(len((self.output / "labels/train/10.txt").read_text().splitlines()), 1)

    def test_mismatched_categories_fail_without_output(self):
        data = dict(self.data)
        data["categories"] = [{"id": 2, "name": "different"}]
        self.val.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "category IDs and names"):
            self.convert()
        self.assertFalse(self.output.exists())

    def test_unknown_reference_and_outside_image_fail_without_output(self):
        self.data["annotations"][0]["image_id"] = 99
        self.write_data()
        with self.assertRaisesRegex(ValueError, "unknown image id"):
            self.convert()
        self.assertFalse(self.output.exists())
        self.data["annotations"][0]["image_id"] = 10
        self.data["images"][0]["file_name"] = "../train.json"
        self.write_data()
        with self.assertRaisesRegex(ValueError, "outside its image root"):
            self.convert()
        self.assertFalse(self.output.exists())

    def test_invalid_box_and_existing_destination_are_rejected(self):
        self.data["annotations"][0]["bbox"] = [101, 0, 5, 5]
        self.write_data()
        with self.assertRaisesRegex(ValueError, "does not overlap"):
            self.convert()
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        (self.output / "keep").write_text("untouched")
        with self.assertRaises(FileExistsError):
            self.convert()
        self.assertEqual((self.output / "keep").read_text(), "untouched")


if __name__ == "__main__":
    unittest.main()
