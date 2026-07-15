import pytest
from rclpy.qos import ReliabilityPolicy

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
