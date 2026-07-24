"""Small NumPy/OpenCV inference wrapper for NanoDet-Plus ONNX models."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
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

DEFAULT_INPUT_SIZE = (320, 320)
DEFAULT_STRIDES = (8, 16, 32, 64)
DEFAULT_REG_MAX = 7
DEFAULT_MEAN = (103.53, 116.28, 123.675)
DEFAULT_STD = (57.375, 57.12, 58.395)


@dataclass(frozen=True)
class NanoDetModelConfig:
    """Validated model geometry and preprocessing configuration."""

    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE
    class_names: tuple[str, ...] = COCO_CLASS_NAMES
    strides: tuple[int, ...] = DEFAULT_STRIDES
    reg_max: int = DEFAULT_REG_MAX
    mean: tuple[float, float, float] = DEFAULT_MEAN
    std: tuple[float, float, float] = DEFAULT_STD
    metadata_path: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed_length_values(
    document: dict,
    key: str,
    default: Sequence,
    length: int,
    value_type,
) -> tuple:
    values = document.get(key, default)
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"Model metadata {key} must contain {length} values")
    return tuple(value_type(value) for value in values)


def load_model_config(
    model_path: str, metadata_path: str = ""
) -> NanoDetModelConfig:
    """Load an explicit or adjacent metadata sidecar, or use COCO defaults."""

    model = Path(model_path).expanduser().resolve()
    explicit_metadata = metadata_path.strip()
    metadata = (
        Path(explicit_metadata).expanduser().resolve()
        if explicit_metadata
        else model.with_suffix(".metadata.json")
    )
    if not metadata.is_file():
        if explicit_metadata:
            raise FileNotFoundError(f"NanoDet metadata not found: {metadata}")
        return NanoDetModelConfig()

    try:
        document = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not read NanoDet metadata {metadata}: {error}"
        ) from error
    if not isinstance(document, dict):
        raise ValueError("NanoDet metadata must be a JSON object")
    if document.get("input_layout", "NCHW") != "NCHW":
        raise ValueError("NanoDet metadata input_layout must be NCHW")
    if document.get("input_color", "BGR") != "BGR":
        raise ValueError("NanoDet metadata input_color must be BGR")

    input_size = _fixed_length_values(
        document, "input_size", DEFAULT_INPUT_SIZE, 2, int
    )
    if any(value <= 0 for value in input_size):
        raise ValueError("NanoDet metadata input_size values must be positive")
    class_values = document.get("class_names", COCO_CLASS_NAMES)
    if not isinstance(class_values, (list, tuple)) or not class_values:
        raise ValueError("NanoDet metadata class_names must be a non-empty list")
    class_names = tuple(str(value).strip() for value in class_values)
    if any(not value for value in class_names) or len(set(class_names)) != len(
        class_names
    ):
        raise ValueError("NanoDet metadata class_names must be non-empty and unique")
    stride_values = document.get("strides", DEFAULT_STRIDES)
    if not isinstance(stride_values, (list, tuple)) or not stride_values:
        raise ValueError("NanoDet metadata strides must be a non-empty list")
    strides = tuple(int(value) for value in stride_values)
    if any(value <= 0 for value in strides):
        raise ValueError("NanoDet metadata strides must be positive")
    reg_max = int(document.get("reg_max", DEFAULT_REG_MAX))
    if reg_max < 0:
        raise ValueError("NanoDet metadata reg_max must be non-negative")
    mean = _fixed_length_values(document, "mean", DEFAULT_MEAN, 3, float)
    std = _fixed_length_values(document, "std", DEFAULT_STD, 3, float)
    if any(value == 0.0 for value in std):
        raise ValueError("NanoDet metadata std values must be non-zero")

    expected_priors = sum(
        math.ceil(input_size[0] / stride) * math.ceil(input_size[1] / stride)
        for stride in strides
    )
    expected_channels = len(class_names) + 4 * (reg_max + 1)
    expected_output_shape = (1, expected_priors, expected_channels)
    declared_output_shape = document.get("output_shape")
    if declared_output_shape is not None:
        declared_output_shape = tuple(int(value) for value in declared_output_shape)
        if declared_output_shape != expected_output_shape:
            raise ValueError(
                "NanoDet metadata output_shape does not match its classes and "
                "geometry: "
                f"expected {expected_output_shape}, got {declared_output_shape}"
            )

    declared_digest = str(document.get("sha256", "")).strip().lower()
    if declared_digest:
        actual_digest = _sha256(model)
        if declared_digest != actual_digest:
            raise ValueError(
                f"NanoDet model SHA-256 mismatch for {model}: "
                f"expected {declared_digest}, got {actual_digest}"
            )

    return NanoDetModelConfig(
        input_size=input_size,
        class_names=class_names,
        strides=strides,
        reg_max=reg_max,
        mean=mean,
        std=std,
        metadata_path=str(metadata),
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

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
        score_threshold: float = 0.4,
        nms_threshold: float = 0.6,
        max_detections: int = 100,
        class_names: Sequence[str] = COCO_CLASS_NAMES,
        strides: Sequence[int] = DEFAULT_STRIDES,
        reg_max: int = DEFAULT_REG_MAX,
        mean: Sequence[float] = DEFAULT_MEAN,
        std: Sequence[float] = DEFAULT_STD,
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
        if len(input_size) != 2 or any(int(value) <= 0 for value in input_size):
            raise ValueError("input_size must contain two positive values")
        if not class_names:
            raise ValueError("class_names must not be empty")
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean and std must each contain three values")
        if any(float(value) == 0.0 for value in std):
            raise ValueError("std values must be non-zero")

        self.input_size = tuple(input_size)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.class_names = tuple(class_names)
        self.reg_max = reg_max
        self.priors = generate_center_priors(self.input_size, strides)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
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
                model_inputs = self._session.get_inputs()
                if len(model_inputs) != 1:
                    raise ValueError(
                        f"Expected one NanoDet model input, got {len(model_inputs)}"
                    )
                model_input = model_inputs[0]
                expected_input_shape = (
                    1,
                    3,
                    self.input_size[1],
                    self.input_size[0],
                )
                if all(isinstance(value, int) for value in model_input.shape) and tuple(
                    model_input.shape
                ) != expected_input_shape:
                    raise ValueError(
                        f"Expected NanoDet input {expected_input_shape}, "
                        f"model requires {tuple(model_input.shape)}"
                    )
                model_outputs = self._session.get_outputs()
                expected_output_shape = (
                    1,
                    self.priors.shape[0],
                    len(self.class_names) + 4 * (self.reg_max + 1),
                )
                if (
                    len(model_outputs) == 1
                    and all(isinstance(value, int) for value in model_outputs[0].shape)
                    and tuple(model_outputs[0].shape) != expected_output_shape
                ):
                    raise ValueError(
                        f"Expected NanoDet output {expected_output_shape}, "
                        f"model provides {tuple(model_outputs[0].shape)}"
                    )
                self._input_name = model_input.name
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
        normalized = (resized.astype(np.float32) - self.mean) / self.std
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
