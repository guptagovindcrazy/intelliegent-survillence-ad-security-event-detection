"""
layers/detection_tracking.py
-----------------------------
LAYER 1 — Detection & Tracking

Responsibility:
    * Run RT-DETR (or YOLO, config-selectable) inference on a frame.
    * Associate detections across frames into persistent tracks using ByteTrack.
    * Emit a clean, typed list of TrackedObject records for downstream layers.

Design notes for viva:
    - RT-DETR (Real-Time Detection Transformer) replaces YOLO as the detector per
      teacher feedback: it is a transformer-based, NMS-free detector, so it avoids
      the hand-tuned IoU/NMS post-processing step entirely and tends to hold up
      better in cluttered, overlapping-object scenes than an anchor-free CNN head.
      Ultralytics exposes RT-DETR through the identical Model API as YOLO, so the
      rest of this pipeline (ByteTrack association, output contract) is unchanged.
    - Ultralytics' `model.track()` already wraps ByteTrack (via `bytetrack.yaml`) and
      handles the two-stage association (high-confidence + low-confidence boxes) that
      is ByteTrack's key contribution over SORT/DeepSORT: it recovers occluded /
      low-confidence detections instead of discarding them, which materially reduces
      ID switches.
    - `persist=True` tells Ultralytics to keep the tracker's internal state (Kalman
      filters, track table) alive between calls instead of re-initializing per frame.
    - In the production/multi-camera setup (see src/production/), each camera gets
      its own DetectionTrackingLayer instance -- i.e. its own model load and its own
      tracker state -- so track ID spaces from different cameras never collide.
      This costs more memory than sharing one model across cameras, but is far
      simpler and safer than trying to share Ultralytics' internal tracker state
      across threads, which is not a supported use case.
"""

from dataclasses import dataclass
from typing import List, Tuple

import os

import numpy as np
from ultralytics import RTDETR, YOLO

from src.config import DetectionConfig, TrackerConfig


@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]   # (x1, y1, x2, y2)

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def foot_point(self) -> Tuple[float, float]:
        """Bottom-center of the bbox — a better ground-plane proxy than centroid
        for zone intrusion, since it approximates where the person is standing."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, y2)


class DetectionTrackingLayer:
    def __init__(self, det_cfg: DetectionConfig, track_cfg: TrackerConfig):
        self.det_cfg = det_cfg
        self.track_cfg = track_cfg
        # RT-DETR (transformer-based, NMS-free detection head) replaces YOLO here per
        # teacher feedback -- Ultralytics exposes it through the same Model API, so
        # .track() with a tracker YAML works identically regardless of which class is used.
        if det_cfg.cpu_threads > 0:
            import torch  # local import: only needed when the thread cap is actually used
            torch.set_num_threads(det_cfg.cpu_threads)
        model_cls = RTDETR if det_cfg.model_type == "rtdetr" else YOLO
        self.model = model_cls(det_cfg.weights)
        self._tracker_yaml = self._materialize_tracker_yaml(track_cfg)

    def _materialize_tracker_yaml(self, cfg: TrackerConfig) -> str:
        """Ultralytics reads tracker hyperparameters from a YAML file. We write a
        temp copy so TrackerConfig stays the single source of truth instead of a
        second hand-edited YAML living in the repo."""
        import tempfile
        import yaml

        tracker_dict = {
            "tracker_type": "bytetrack",
            "track_high_thresh": cfg.track_high_thresh,
            "track_low_thresh": cfg.track_low_thresh,
            "new_track_thresh": cfg.new_track_thresh,
            "track_buffer": cfg.track_buffer,
            "match_thresh": cfg.match_thresh,
            "fuse_score": True,
        }
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.safe_dump(tracker_dict, f)
        f.close()
        return f.name

    def close(self):
        """Deletes the temp tracker-hyperparameter YAML written by
        _materialize_tracker_yaml. Every DetectionTrackingLayer instance
        leaks one of these on disk unless this is called -- callers that
        create a short-lived instance (one run of the desktop app, one
        video in a batch) should call this when done; production/live_main.py
        callers create one instance per camera for the whole process
        lifetime, so the leak there is bounded and cleanup is optional.
        Safe to call more than once."""
        path = getattr(self, "_tracker_yaml", None)
        if path:
            try:
                os.remove(path)
            except OSError:
                pass  # already gone, or never existed -- either way, nothing to do
            self._tracker_yaml = None

    def __del__(self):
        # Best-effort fallback for callers that forget to call close()
        # explicitly. Not guaranteed to run (CPython usually does via
        # refcounting, but never rely on __del__ alone for correctness) --
        # this is a safety net, not the primary cleanup mechanism.
        try:
            self.close()
        except Exception:
            pass

    def process(self, frame: np.ndarray) -> List[TrackedObject]:
        """Run detection + tracking on a single BGR frame. Returns persistent-ID objects."""
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.det_cfg.conf_threshold,
            iou=self.det_cfg.iou_threshold,
            classes=self.det_cfg.classes,
            device=self.det_cfg.device,
            imgsz=self.det_cfg.imgsz,
            tracker=self._tracker_yaml,
            verbose=False,
        )

        tracked_objects: List[TrackedObject] = []
        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return tracked_objects  # nothing detected / nothing tracked this frame

        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        for box, tid, conf, cid in zip(boxes, ids, confs, cls_ids):
            tracked_objects.append(
                TrackedObject(
                    track_id=int(tid),
                    class_id=int(cid),
                    class_name=self.model.names.get(int(cid), str(cid)),
                    confidence=float(conf),
                    bbox_xyxy=tuple(box.tolist()),
                )
            )
        return tracked_objects
