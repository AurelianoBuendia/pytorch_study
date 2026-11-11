import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from yolo11 import YOLO
from yolo11.inference import _letterbox, _postprocess
from yolo11.model import load_model


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models/yolo11n.pt"


class UnsafeObject:
    def __reduce__(self):
        return (eval, ("1 + 1",))


class Yolo11InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.yolo = YOLO(WEIGHTS)

    def test_checkpoint_output_matches_reference(self):
        sample = torch.linspace(0, 1, 3 * 64 * 64).reshape(1, 3, 64, 64)
        with torch.inference_mode():
            output = self.yolo.model(sample)
        self.assertEqual(tuple(output.shape), (1, 84, 84))
        indices = [0, 40, 83]
        expected = torch.tensor([
            [4.9334736, 3.4139242, 34.1987534],
            [3.4422455, 40.4525681, 33.6075211],
            [11.3708954, 7.0338097, 64.6292114],
            [6.6976795, 49.0829353, 61.1989403],
        ])
        torch.testing.assert_close(output[0, :4, indices], expected, rtol=2e-4, atol=2e-4)

    def test_rejects_unexpected_pickle_globals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.pt"
            torch.save({"model": UnsafeObject()}, path)
            with self.assertRaisesRegex(pickle.UnpicklingError, "unsupported checkpoint global"):
                load_model(path)

    def test_prediction_api_and_stream(self):
        path = ROOT / "images/couple-3687274_960_720.jpg"
        result = self.yolo.predict(path, imgsz=320)[0]
        self.assertGreaterEqual(len(result.boxes), 2)
        self.assertEqual(result.boxes.xyxy.shape[1], 4)
        self.assertEqual(result.boxes.data.shape[1], 6)
        self.assertIsNone(result.boxes.id)
        self.assertEqual(result.names[0], "person")
        self.assertEqual(result.orig_img.shape[:2], result.orig_shape)
        streamed = self.yolo.predict([path, path], imgsz=320, stream=True)
        self.assertEqual(sum(1 for _ in streamed), 2)

    def test_letterbox_and_class_aware_nms(self):
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        padded, ratio, padding = _letterbox(image, (64, 64))
        self.assertEqual(padded.shape, (64, 64, 3))
        self.assertEqual((ratio, padding), (0.8, (0, 16)))
        # Two coincident boxes with different classes survive class-aware NMS.
        prediction = torch.tensor([[32, 32], [32, 32], [16, 16], [16, 16],
                                   [0.9, 0.1], [0.1, 0.8]], dtype=torch.float32)
        result = _postprocess(prediction, image.shape, ratio, padding, 0.25, 0.7, 300, None, False)
        self.assertEqual(len(result), 2)
        filtered = _postprocess(prediction, image.shape, ratio, padding, 0.25, 0.7, 300, 1, False)
        self.assertEqual(filtered[:, 5].tolist(), [1])
        suppressed = _postprocess(prediction, image.shape, ratio, padding, 0.25, 0.7, 300, None, True)
        self.assertEqual(len(suppressed), 1)

    def test_dynamic_onnx_matches_torch(self):
        try:
            import onnx
            import onnxruntime
        except ImportError as exc:
            self.skipTest(f"ONNX validation dependencies are missing: {exc}")
        with tempfile.TemporaryDirectory() as directory:
            path = self.yolo.export(imgsz=64, dynamic=True, path=Path(directory) / "model.onnx")
            onnx.checker.check_model(onnx.load(path))
            session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
            for shape in ((1, 3, 64, 64), (2, 3, 96, 128)):
                source = np.random.default_rng(42).random(shape, dtype=np.float32)
                with torch.inference_mode():
                    expected = self.yolo.model(torch.from_numpy(source)).numpy()
                actual = session.run(None, {"images": source})[0]
                np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
