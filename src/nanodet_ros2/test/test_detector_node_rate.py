import pytest
from rclpy.qos import ReliabilityPolicy

from nanodet_ros2.nanodet_engine import Detection
from nanodet_ros2.detector_node import NanoDetNode, sensor_qos


def test_rate_limited_processing_uses_latest_frame_once():
    node = NanoDetNode.__new__(NanoDetNode)
    node._latest_image_message = None
    processed = []
    node._process_image = processed.append

    node._store_latest_image("older")
    node._store_latest_image("newest")
    node._process_latest_image()
    node._process_latest_image()

    assert processed == ["newest"]


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
