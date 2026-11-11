"""ByteTrack association with a constant-velocity box Kalman filter."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TrackerConfig:
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True


def load_config(path="bytetrack.yaml"):
    path = Path(path)
    if str(path) == "bytetrack.yaml" and not path.exists():
        path = Path(__file__).with_name("bytetrack.yaml")
    with path.open(encoding="utf-8") as file:
        values = yaml.safe_load(file)
    if not isinstance(values, dict):
        raise ValueError(f"{path}: tracker configuration must be a mapping")
    values = values.copy()
    if values.pop("tracker_type", None) != "bytetrack":
        raise ValueError(f"{path}: tracker_type must be bytetrack")
    unknown = set(values) - set(TrackerConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"{path}: unsupported tracker options: {', '.join(sorted(unknown))}")
    config = TrackerConfig(**values)
    for key in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"):
        value = getattr(config, key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be a number between 0 and 1")
    if config.track_low_thresh >= config.track_high_thresh:
        raise ValueError("track_low_thresh must be less than track_high_thresh")
    if isinstance(config.track_buffer, bool) or not isinstance(config.track_buffer, int) or config.track_buffer < 0:
        raise ValueError("track_buffer must be a nonnegative integer")
    if not isinstance(config.fuse_score, bool):
        raise ValueError("fuse_score must be a boolean")
    return config


def _xyah(box):
    width, height = box[2] - box[0], box[3] - box[1]
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, width / height, height])


class _Kalman:
    def __init__(self):
        self.motion = np.eye(8)
        self.motion[:4, 4:] = np.eye(4)
        self.observation = np.eye(4, 8)

    @staticmethod
    def _noise(height, position, velocity):
        return np.square([position * height, position * height, 1e-2, position * height,
                          velocity * height, velocity * height, 1e-5, velocity * height])

    def initiate(self, box):
        mean = np.r_[_xyah(box), np.zeros(4)]
        covariance = np.diag(self._noise(mean[3], 2 / 20, 10 / 160))
        covariance[2, 2] = 1e-4
        return mean, covariance

    def predict(self, mean, covariance):
        noise = np.diag(self._noise(mean[3], 1 / 20, 1 / 160))
        mean = self.motion @ mean
        return mean, self.motion @ covariance @ self.motion.T + noise

    def update(self, mean, covariance, box):
        projected = self.observation @ mean
        noise = np.diag(np.square([mean[3] / 20, mean[3] / 20, 0.1, mean[3] / 20]))
        innovation = self.observation @ covariance @ self.observation.T + noise
        gain = np.linalg.solve(innovation, self.observation @ covariance).T
        mean = mean + gain @ (_xyah(box) - projected)
        covariance = covariance - gain @ innovation @ gain.T
        return mean, covariance


class _Track:
    def __init__(self, detection, track_id, frame, kalman):
        self.mean, self.covariance = kalman.initiate(detection[:4])
        self.id = track_id
        self.score = detection[4]
        self.cls = detection[5]
        self.last_frame = frame
        self.start_frame = frame
        self.confirmed = frame == 1
        self.state = "tracked"

    @property
    def box(self):
        x, y, aspect, height = self.mean[:4]
        width = aspect * height
        return np.array([x - width / 2, y - height / 2, x + width / 2, y + height / 2])

    def predict(self, kalman):
        if self.state != "tracked":
            self.mean[7] = 0
        self.mean, self.covariance = kalman.predict(self.mean, self.covariance)

    def update(self, detection, frame, kalman):
        self.mean, self.covariance = kalman.update(self.mean, self.covariance, detection[:4])
        self.score = detection[4]
        self.cls = detection[5]
        self.last_frame = frame
        self.state = "tracked"
        self.confirmed = True


def _iou(tracks, detections):
    if not tracks or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))
    boxes = np.stack([track.box for track in tracks])
    target = detections[:, :4]
    top_left = np.maximum(boxes[:, None, :2], target[None, :, :2])
    bottom_right = np.minimum(boxes[:, None, 2:], target[None, :, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0), axis=2)
    area = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    target_area = np.prod(np.maximum(target[:, 2:] - target[:, :2], 0), axis=1)
    union = area[:, None] + target_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _assign(tracks, detections, threshold, fuse_score):
    if not tracks or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))
    similarity = _iou(tracks, detections)
    if fuse_score:
        similarity *= detections[None, :, 4]
    cost = 1 - similarity
    # Allow unmatched rows and columns at the configured cost limit.
    rows, cols = cost.shape
    augmented = np.full((rows + cols, rows + cols), 1e6)
    augmented[:rows, :cols] = np.where(cost <= threshold, cost, 1e6)
    augmented[np.arange(rows), cols + np.arange(rows)] = threshold / 2 + 1e-7
    augmented[rows + np.arange(cols), np.arange(cols)] = threshold / 2 + 1e-7
    augmented[rows:, cols:] = 0
    assignment_rows, assignment_cols = linear_sum_assignment(augmented)
    matches = [(r, c) for r, c in zip(assignment_rows, assignment_cols) if r < rows and c < cols]
    matched_rows = {r for r, _ in matches}
    matched_cols = {c for _, c in matches}
    return matches, [r for r in range(rows) if r not in matched_rows], [
        c for c in range(cols) if c not in matched_cols
    ]


class BYTETracker:
    """Track detections across frames; IDs are local to this instance."""

    def __init__(self, config, frame_rate=30):
        if not np.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("frame_rate must be positive and finite")
        self.config = config
        self.frame_rate = frame_rate
        self.max_lost = int(config.track_buffer * frame_rate / 30)
        self.kalman = _Kalman()
        self.tracks = []
        self.frame = 0
        self.next_id = 1

    def update(self, detections):
        detections = np.asarray(detections, dtype=np.float64)
        if detections.ndim != 2 or detections.shape[1] != 6:
            raise ValueError("detections must have shape (N, 6): xyxy, confidence, class")
        if (not np.isfinite(detections).all() or
                np.any(detections[:, 2:4] <= detections[:, :2]) or
                np.any((detections[:, 4] < 0) | (detections[:, 4] > 1))):
            raise ValueError("detections must contain finite, positive-area boxes and confidence in [0, 1]")
        self.frame += 1
        high = detections[detections[:, 4] >= self.config.track_high_thresh]
        low = detections[(detections[:, 4] > self.config.track_low_thresh) &
                         (detections[:, 4] < self.config.track_high_thresh)]
        confirmed = [t for t in self.tracks if t.confirmed and t.state in ("tracked", "lost")]
        tentative = [t for t in self.tracks if not t.confirmed and t.state == "tracked"]
        for track in confirmed:
            track.predict(self.kalman)

        matches, unmatched, remaining = _assign(confirmed, high, self.config.match_thresh, self.config.fuse_score)
        for row, col in matches:
            confirmed[row].update(high[col], self.frame, self.kalman)
        unmatched_active = [confirmed[i] for i in unmatched if confirmed[i].state == "tracked"]
        matches, unmatched_active_indices, _ = _assign(unmatched_active, low, 0.5, False)
        for row, col in matches:
            unmatched_active[row].update(low[col], self.frame, self.kalman)
        for index in unmatched_active_indices:
            unmatched_active[index].state = "lost"

        remaining_high = high[remaining]
        matches, unmatched_tentative, remaining = _assign(tentative, remaining_high, 0.7, self.config.fuse_score)
        for row, col in matches:
            tentative[row].update(remaining_high[col], self.frame, self.kalman)
        for index in unmatched_tentative:
            tentative[index].state = "removed"
        for index in remaining:
            detection = remaining_high[index]
            if detection[4] >= self.config.new_track_thresh:
                self.tracks.append(_Track(detection, self.next_id, self.frame, self.kalman))
                self.next_id += 1

        for track in self.tracks:
            if track.state == "lost" and self.frame - track.last_frame > self.max_lost:
                track.state = "removed"
        tracked = [t for t in self.tracks if t.state == "tracked" and t.confirmed]
        lost = [t for t in self.tracks if t.state == "lost"]
        if tracked and lost:
            lost_boxes = np.stack([np.r_[t.box, 0, 0] for t in lost])
            rows, cols = np.where(_iou(tracked, lost_boxes) > 0.85)
            for row, col in zip(rows, cols):
                current, previous = tracked[row], lost[col]
                if current.last_frame - current.start_frame > previous.last_frame - previous.start_frame:
                    previous.state = "removed"
                else:
                    current.state = "removed"
        self.tracks = [t for t in self.tracks if t.state != "removed"]
        active = [t for t in self.tracks if t.state == "tracked" and t.confirmed]
        return np.array(
            [np.r_[track.box, track.id, track.score, track.cls] for track in active], dtype=np.float32
        ).reshape(-1, 7)
