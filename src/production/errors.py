"""
production/errors.py
----------------------
Structured error taxonomy for the live/production pipeline.

Design notes for viva:
    Instead of letting exceptions propagate as raw stack traces (which crash a
    worker thread silently or dump unreadable output to a log), every failure
    mode this system anticipates is represented as a SurveillanceError: a
    small, structured object with a stable code, a human message, a
    timestamp, and a `recoverable` flag. This makes failures:
      - loggable in a consistent, greppable format
      - displayable in a UI without string-parsing an exception message
      - actionable by a fallback/retry chain that only needs to check
        `err.recoverable`, not the error's type or message text
"""

import time
from dataclasses import dataclass, field


@dataclass
class SurveillanceError:
    code: str
    message: str
    camera_id: str = "unknown"
    recoverable: bool = True
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "camera_id": self.camera_id,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        tag = "recoverable" if self.recoverable else "FATAL"
        return f"[{self.code}] ({tag}) camera={self.camera_id}: {self.message}"


# ---------------------------------------------------------------------------
# Standard error codes used across the production pipeline
# ---------------------------------------------------------------------------
def cam_not_found(camera_id: str, source: str) -> SurveillanceError:
    return SurveillanceError(
        code="ERR_CAM_404",
        message=f"Camera/stream not found or failed to open: {source}",
        camera_id=camera_id,
        recoverable=True,  # retry on a delay -- the stream may come back
    )


def cam_permission_denied(camera_id: str, source: str) -> SurveillanceError:
    return SurveillanceError(
        code="ERR_CAM_403",
        message=f"Permission denied opening stream: {source}",
        camera_id=camera_id,
        recoverable=False,  # retrying won't fix a credentials/permission problem
    )


def model_load_failed(camera_id: str, detail: str) -> SurveillanceError:
    return SurveillanceError(
        code="ERR_MODEL_500",
        message=f"Detection model failed to load: {detail}",
        camera_id=camera_id,
        recoverable=True,  # caller retries model_load_retries times before giving up
    )


def gpu_unavailable(camera_id: str) -> SurveillanceError:
    return SurveillanceError(
        code="ERR_GPU_UNAVAILABLE",
        message="CUDA GPU not available -- falling back to CPU inference",
        camera_id=camera_id,
        recoverable=True,  # this is a graceful-degradation event, not a hard failure
    )


def zone_config_invalid(camera_id: str, detail: str) -> SurveillanceError:
    return SurveillanceError(
        code="ERR_ZONE_CONFIG_400",
        message=f"Malformed zone/tripwire configuration: {detail}",
        camera_id=camera_id,
        recoverable=False,  # bad config needs a human fix, not a retry
    )


def stream_timeout(camera_id: str, seconds: float) -> SurveillanceError:
    return SurveillanceError(
        code="ERR_STREAM_TIMEOUT",
        message=f"No frames received for {seconds:.0f}s -- stream considered dead",
        camera_id=camera_id,
        recoverable=True,  # camera worker will attempt a reconnect
    )


class RetryExhausted(Exception):
    """Raised when a recoverable error's retry budget is exhausted."""
    def __init__(self, last_error: SurveillanceError):
        self.last_error = last_error
        super().__init__(str(last_error))


def retry_with_fallback(fn, retries: int, on_failure, *args, **kwargs):
    """Call fn(*args, **kwargs) up to `retries` times, calling on_failure(attempt,
    exception) between attempts. Raises RetryExhausted with the final error if
    every attempt fails. Used e.g. for model loading (ERR_MODEL_500)."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- intentionally broad: any load failure counts
            last_exc = exc
            on_failure(attempt, exc)
    raise RetryExhausted(model_load_failed("unknown", str(last_exc)))
