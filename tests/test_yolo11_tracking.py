import tempfile
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import cv2
import numpy as np
import torch

from yolo11 import YOLO
from yolo11.byte_tracker import BYTETracker, TrackerConfig, _assign, load_config
from yolo11.inference import Results


ROOT = Path(__file__).resolve().parents[1]
EMPTY = np.empty((0, 6), dtype=np.float32)


def detection(x=10, score=0.9, cls=0):
    return np.array([[x, 10, x + 20, 40, score, cls]], dtype=np.float32)


class TrackingTests(unittest.TestCase):
    def test_high_low_lost_reactivation_and_expiry(self):
        tracker = BYTETracker(TrackerConfig(track_buffer=2, fuse_score=False))
        self.assertEqual(tracker.update(detection())[0, 4], 1)
        self.assertEqual(tracker.update(detection(11, 0.15))[0, 4], 1)
        self.assertEqual(tracker.update(EMPTY).shape, (0, 7))
        self.assertEqual(tracker.update(detection(12))[0, 4], 1)
        tracker.update(EMPTY)
        tracker.update(EMPTY)
        tracker.update(EMPTY)
        self.assertEqual(tracker.update(detection(12)).shape, (0, 7))
        self.assertEqual(tracker.update(detection(12))[0, 4], 2)

    def test_low_detections_do_not_create_tracks_and_late_track_needs_confirmation(self):
        tracker = BYTETracker(TrackerConfig(fuse_score=False))
        self.assertEqual(len(tracker.update(detection(score=0.15))), 0)
        self.assertEqual(len(tracker.update(detection())), 0)
        self.assertEqual(tracker.update(detection())[0, 4], 1)
        tracker.update(detection(100))
        result = tracker.update(detection(100))
        self.assertEqual(result[0, 4], 2)

    def test_unmatched_tentative_track_removed(self):
        tracker = BYTETracker(TrackerConfig(fuse_score=False))
        tracker.update(EMPTY)
        tracker.update(detection())
        tracker.update(EMPTY)
        self.assertEqual(len(tracker.tracks), 0)

    def test_assignment_accounts_for_unmatched_choices(self):
        tracks = [object(), object()]
        detections = np.zeros((2, 6))
        with patch("yolo11.byte_tracker._iou", return_value=np.array([[0.7, 0.6], [0.6, 0.0]])):
            matches, rows, cols = _assign(tracks, detections, 0.4, False)
        self.assertEqual(matches, [(0, 0)])
        self.assertEqual((rows, cols), ([1], [1]))

    def test_config_validation_and_missing_file(self):
        config = load_config()
        self.assertEqual(config.track_buffer, 30)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bytetrack.yaml"
            path.write_text("tracker_type: bytetrack\ntrack_low_thresh: 0.5\ntrack_high_thresh: 0.3\n")
            with self.assertRaisesRegex(ValueError, "track_low_thresh"):
                load_config(path)
            path.write_text("tracker_type: botsort\n")
            with self.assertRaisesRegex(ValueError, "tracker_type"):
                load_config(path)
            path.write_text("tracker_type: bytetrack\nunknown_option: 1\n")
            with self.assertRaisesRegex(ValueError, "unsupported tracker options"):
                load_config(path)
            with self.assertRaises(FileNotFoundError):
                load_config(path.parent / "missing.yaml")

    def test_invalid_detections_and_frame_rate(self):
        tracker = BYTETracker(TrackerConfig())
        for value in (np.array([1, 2]), detection(score=2), np.array([[0, 0, 0, 3, 0.5, 0]])):
            with self.assertRaises(ValueError):
                tracker.update(value)
        with self.assertRaises(ValueError):
            BYTETracker(TrackerConfig(), 0)


class TrackingApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = YOLO(ROOT / "models/yolo11n.pt")

    def setUp(self):
        self.model.reset_tracker()

    def test_real_model_on_repeated_frames(self):
        image = cv2.imread(str(ROOT / "images/couple-3687274_960_720.jpg"))
        results = self.model.track([image, image], imgsz=320, stream=True)
        first, second = list(results)
        self.assertGreaterEqual(len(first.boxes), 2)
        self.assertTrue(first.boxes.is_track)
        self.assertEqual(first.boxes.data.shape[1], 7)
        self.assertEqual(first.boxes.id.tolist(), second.boxes.id.tolist())
        self.assertTrue(all(int(i) >= 1 for i in first.boxes.id))
        self.assertEqual(len(first.boxes.conf), len(first.boxes))

    def test_persistent_frames_reset_and_config_guard(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        def fake_predict(model, source, *args):
            return Results(source, None, model.names, torch.tensor(detection()))
        with patch("yolo11.api.predict_image", side_effect=fake_predict):
            first = self.model.track(frame, persist=True)[0]
            second = self.model.track(frame, persist=True)[0]
            self.assertEqual(first.boxes.id.tolist(), second.boxes.id.tolist())
            with self.assertRaisesRegex(ValueError, "same tracker configuration"):
                self.model.track(frame, persist=True, frame_rate=25)
            self.model.reset_tracker()
            self.assertEqual(self.model.track(frame, persist=True)[0].boxes.id.tolist(), [1])
            self.assertEqual(self.model.track(frame)[0].boxes.id.tolist(), [1])

    def test_video_stream_and_resource_release(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "sample.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
            self.assertTrue(writer.isOpened())
            for value in (0, 30, 60):
                writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
            writer.release()
            def fake_predict(model, source, *args):
                return Results(source, None, model.names, torch.tensor(detection()))
            with patch("yolo11.api.predict_image", side_effect=fake_predict):
                results = list(self.model.track(path, stream=True))
                self.assertEqual(len(results), 3)
                self.assertTrue(all(result.path == path for result in results))
                self.assertTrue(all(result.boxes.id.tolist() == [1] for result in results))
                stream = self.model.track(path, stream=True)
                next(stream)
                with self.assertRaises(RuntimeError):
                    self.model.track(path)
                stream.close()
                self.assertEqual(len(self.model.track(path)), 3)
            with self.assertRaisesRegex(ValueError, "could not open video"):
                self.model.track(str(Path(directory) / "missing.mp4"))

    def test_video_example_end_to_end(self):
        from yolo11.track_video import main

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.avi"
            output = Path(directory) / "output.mp4"
            image = cv2.imread(str(ROOT / "images/couple-3687274_960_720.jpg"))
            image = cv2.resize(image, (320, 224))
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 224))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(image)
            writer.release()
            argv = ["track_video", str(source), str(output), "--weights",
                    str(ROOT / "models/yolo11n.pt"), "--imgsz", "320"]
            with mock.patch("sys.argv", argv):
                main()
            capture = cv2.VideoCapture(str(output))
            try:
                self.assertTrue(capture.isOpened())
                count = 0
                while capture.read()[0]:
                    count += 1
                self.assertEqual(count, 3)
            finally:
                capture.release()


if __name__ == "__main__":
    unittest.main()
