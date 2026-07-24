import pytest
from rclpy.qos import ReliabilityPolicy

from nanodet_ros2.nanodet_engine import Detection
from nanodet_ros2.detector_node import NanoDetNode, sensor_qos


def test_rate_limited_processing_uses_latest_frame_once():
    node = NanoDetNode.__new__(NanoDetNode)
    node._latest_image_message = None
    node._latest_cv_image = None
    processed = []
    node._publish_live_annotation = lambda message: None
    node._process_image = (
        lambda message, image=None: processed.append((message, image))
    )

    node._store_latest_image("older")
    node._store_latest_image("newest")
    node._process_latest_image()
    node._process_latest_image()

    assert processed == [("newest", None)]


def test_live_annotation_publishes_each_input_but_infers_only_newest():
    node = NanoDetNode.__new__(NanoDetNode)
    node._latest_image_message = None
    node._latest_cv_image = None
    published = []
    processed = []
    node._publish_live_annotation = (
        lambda message: published.append(message) or f"image-{message}"
    )
    node._process_image = (
        lambda message, image=None: processed.append((message, image))
    )

    node._store_latest_image("older")
    node._store_latest_image("newest")
    node._process_latest_image()

    assert published == ["older", "newest"]
    assert processed == [("newest", "image-newest")]


def test_live_annotation_scales_cached_detections_for_new_resolution():
    node = NanoDetNode.__new__(NanoDetNode)
    node._latest_detection_image_size = (320, 240)
    node._latest_detections = (
        Detection(1, "pallet", 0.8, (32.0, 24.0, 160.0, 120.0)),
    )

    assert node._detections_for_image(640, 480) == (
        Detection(1, "pallet", 0.8, (64.0, 48.0, 320.0, 240.0)),
    )


def test_sensor_qos_supports_explicit_reliability():
    assert sensor_qos(reliability="best_effort").reliability == (
        ReliabilityPolicy.BEST_EFFORT
    )
    assert sensor_qos(reliability="reliable").reliability == (
        ReliabilityPolicy.RELIABLE
    )
    with pytest.raises(ValueError, match="best_effort or reliable"):
        sensor_qos(reliability="sometimes")


def test_allowed_labels_filter_keeps_only_person():
    node = NanoDetNode.__new__(NanoDetNode)
    node._allowed_labels = NanoDetNode._parse_allowed_labels("person")

    detections = (
        Detection(0, "person", 0.9, (0.0, 0.0, 1.0, 1.0)),
        Detection(2, "car", 0.8, (0.0, 0.0, 1.0, 1.0)),
    )

    assert node._filter_detections(detections) == (
        Detection(0, "person", 0.9, (0.0, 0.0, 1.0, 1.0)),
    )


def test_parse_allowed_labels_accepts_commas_and_spaces():
    assert NanoDetNode._parse_allowed_labels("person, car bicycle") == frozenset(
        {"person", "car", "bicycle"}
    )
