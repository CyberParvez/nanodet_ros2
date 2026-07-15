"""Small NumPy/OpenCV inference wrapper for NanoDet-Plus ONNX models."""

from dataclasses import dataclass
import math
import time
from typing import Sequence

import cv2
import numpy as np


COCO_CLASS_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic_light", "fire_hydrant", "stop_sign",
    "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports_ball", "kite", "baseball_bat", "baseball_glove", "skateboard",
    "surfboard", "tennis_racket", "bottle", "wine_glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", "chair",
    "couch", "potted_plant", "bed", "dining_table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell_phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy_bear", "hair_drier", "toothbrush",
)


@dataclass(frozen=True)
class Detection:
    """One image-space object detection."""

    class_id: int
    label: str
    score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class InferenceResult:
    """Detections plus the measured model and post-processing duration."""

    detections: tuple[Detection, ...]
    inference_ms: float


def generate_center_priors(
    input_size: tuple[int, int], strides: Sequence[int]
) -> np.ndarray:
    """Generate NanoDet feature-grid origins as (x, y, stride, stride)."""

    input_width, input_height = input_size
    levels = []
    for stride in strides:
        feature_width = math.ceil(input_width / stride)
        feature_height = math.ceil(input_height / stride)
        x_range = np.arange(feature_width, dtype=np.float32) * stride
        y_range = np.arange(feature_height, dtype=np.float32) * stride
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        level_strides = np.full(grid_x.size, stride, dtype=np.float32)
        levels.append(
            np.stack(
                (
                    grid_x.reshape(-1),
                    grid_y.reshape(-1),
                    level_strides,
                    level_strides,
                ),
                axis=1,
            )
        )
    return np.concatenate(levels, axis=0)


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def _hard_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    max_detections: int,
) -> list[int]:
    """Return indices kept by standard greedy non-maximum suppression."""

    if boxes.size == 0:
        return []

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]
    kept = []

    while order.size and len(kept) < max_detections:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break

        remaining = order[1:]
        intersect_x1 = np.maximum(x1[current], x1[remaining])
        intersect_y1 = np.maximum(y1[current], y1[remaining])
        intersect_x2 = np.minimum(x2[current], x2[remaining])
        intersect_y2 = np.minimum(y2[current], y2[remaining])
        intersect_width = np.maximum(0.0, intersect_x2 - intersect_x1)
        intersect_height = np.maximum(0.0, intersect_y2 - intersect_y1)
        intersection = intersect_width * intersect_height
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0.0,
        )
        order = remaining[iou <= iou_threshold]

    return kept


def decode_predictions(
    predictions: np.ndarray,
    priors: np.ndarray,
    original_size: tuple[int, int],
    input_size: tuple[int, int],
    class_names: Sequence[str],
    score_threshold: float,
    nms_threshold: float,
    max_detections: int,
    reg_max: int = 7,
) -> tuple[Detection, ...]:
    """Decode one NanoDet output tensor into original-image coordinates."""

    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]
    if predictions.ndim != 2:
        raise ValueError(f"Expected a 2D prediction tensor, got {predictions.shape}")

    num_classes = len(class_names)
    expected_channels = num_classes + 4 * (reg_max + 1)
    if predictions.shape[1] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} output channels, got {predictions.shape[1]}"
        )
    if predictions.shape[0] != priors.shape[0]:
        raise ValueError(
            f"Model produced {predictions.shape[0]} priors, expected {priors.shape[0]}"
        )

    class_scores = predictions[:, :num_classes]
    labels = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), labels]
    candidate_indices = np.flatnonzero(scores >= score_threshold)
    if candidate_indices.size == 0:
        return ()

    labels = labels[candidate_indices]
    scores = scores[candidate_indices]
    selected_priors = priors[candidate_indices]
    regression = predictions[candidate_indices, num_classes:].reshape(
        -1, 4, reg_max + 1
    )
    distribution = _softmax(regression)
    projection = np.arange(reg_max + 1, dtype=np.float32)
    distances = np.sum(distribution * projection, axis=2)
    distances *= selected_priors[:, 2:3]

    centers = selected_priors[:, :2]
    boxes = np.column_stack(
        (
            centers[:, 0] - distances[:, 0],
            centers[:, 1] - distances[:, 1],
            centers[:, 0] + distances[:, 2],
            centers[:, 1] + distances[:, 3],
        )
    )

    input_width, input_height = input_size
    original_width, original_height = original_size
    boxes[:, (0, 2)] *= original_width / input_width
    boxes[:, (1, 3)] *= original_height / input_height
    boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0.0, original_width - 1.0)
    boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0.0, original_height - 1.0)

    kept_global = []
    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        class_kept = _hard_nms(
            boxes[class_indices],
            scores[class_indices],
            nms_threshold,
            max_detections,
        )
        kept_global.extend(class_indices[class_kept].tolist())

    kept_global.sort(key=lambda index: float(scores[index]), reverse=True)
    kept_global = kept_global[:max_detections]
    return tuple(
        Detection(
            class_id=int(labels[index]),
            label=str(class_names[int(labels[index])]),
            score=float(scores[index]),
            box=tuple(float(value) for value in boxes[index]),
        )
        for index in kept_global
    )


class NanoDet:
    """NanoDet-Plus inference using ONNX Runtime or an OpenCV fallback."""

    MEAN = np.asarray((103.53, 116.28, 123.675), dtype=np.float32)
    STD = np.asarray((57.375, 57.12, 58.395), dtype=np.float32)

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (320, 320),
        score_threshold: float = 0.4,
        nms_threshold: float = 0.6,
        max_detections: int = 100,
        class_names: Sequence[str] = COCO_CLASS_NAMES,
        strides: Sequence[int] = (8, 16, 32, 64),
        reg_max: int = 7,
        runtime: str = "auto",
        runtime_threads: int = 4,
        runtime_allow_spinning: bool = False,
    ) -> None:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1")
        if not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be between 0 and 1")
        if max_detections <= 0:
            raise ValueError("max_detections must be positive")
        if runtime not in ("auto", "onnxruntime", "opencv"):
            raise ValueError("runtime must be auto, onnxruntime, or opencv")
        if runtime_threads < 0:
            raise ValueError("runtime_threads must be non-negative")

        self.input_size = tuple(input_size)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.class_names = tuple(class_names)
        self.reg_max = reg_max
        self.priors = generate_center_priors(self.input_size, strides)
        self.runtime_name = ""
        self._session = None
        self._input_name = ""
        self._net = None

        if runtime in ("auto", "onnxruntime"):
            try:
                import onnxruntime as ort
            except ImportError:
                if runtime == "onnxruntime":
                    raise RuntimeError(
                        "ONNX Runtime was requested but is not installed"
                    ) from None
            else:
                options = ort.SessionOptions()
                options.intra_op_num_threads = runtime_threads
                options.inter_op_num_threads = 1
                allow_spinning = "1" if runtime_allow_spinning else "0"
                options.add_session_config_entry(
                    "session.intra_op.allow_spinning", allow_spinning
                )
                options.add_session_config_entry(
                    "session.inter_op.allow_spinning", allow_spinning
                )
                options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                options.log_severity_level = 3
                self._session = ort.InferenceSession(
                    model_path,
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._session.get_inputs()[0].name
                self.runtime_name = "onnxruntime"

        if self._session is None:
            cv2.setNumThreads(runtime_threads)
            self._net = cv2.dnn.readNetFromONNX(model_path)
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.runtime_name = "opencv"

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize and normalize a BGR image according to the official config."""

        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a non-empty BGR image with three channels")
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) - self.MEAN) / self.STD
        return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])

    def detect(self, image: np.ndarray) -> InferenceResult:
        """Run detection and return boxes in the input image's coordinates."""

        blob = self.preprocess(image)
        started = time.perf_counter()
        if self._session is not None:
            predictions = self._session.run(None, {self._input_name: blob})[0]
        else:
            self._net.setInput(blob)
            predictions = self._net.forward()
        detections = decode_predictions(
            predictions=predictions,
            priors=self.priors,
            original_size=(image.shape[1], image.shape[0]),
            input_size=self.input_size,
            class_names=self.class_names,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            max_detections=self.max_detections,
            reg_max=self.reg_max,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return InferenceResult(detections=detections, inference_ms=elapsed_ms)


# Kept as a compatibility alias for code written before the runtime was selectable.
NanoDetOpenCV = NanoDet
