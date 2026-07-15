#!/usr/bin/env python3
"""Temporary ROS 2 publisher for an OpenCV/V4L2 laptop camera."""

import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image


def camera_qos(reliability: str) -> QoSProfile:
    """Use sensor-data QoS while retaining only the newest camera frame."""
    reliability_policies = {
        'best_effort': ReliabilityPolicy.BEST_EFFORT,
        'reliable': ReliabilityPolicy.RELIABLE,
    }
    normalized_reliability = reliability.strip().lower()
    if normalized_reliability not in reliability_policies:
        raise ValueError('reliability must be best_effort or reliable')

    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=reliability_policies[normalized_reliability],
        durability=DurabilityPolicy.VOLATILE,
    )


class LaptopCameraPublisher(Node):
    """Publish frames from a laptop camera as sensor_msgs/Image."""

    def __init__(self) -> None:
        super().__init__('laptop_camera_publisher')
        self.declare_parameter('device', '/dev/video0')
        self.declare_parameter('topic', '/camera/image_raw')
        self.declare_parameter('frame_id', 'laptop_camera_optical_frame')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 15.0)
        self.declare_parameter('reliability', 'best_effort')

        self._device = str(self.get_parameter('device').value)
        self._topic = str(self.get_parameter('topic').value)
        self._frame_id = str(self.get_parameter('frame_id').value)
        self._requested_width = int(self.get_parameter('width').value)
        self._requested_height = int(self.get_parameter('height').value)
        self._requested_fps = float(self.get_parameter('fps').value)
        self._reliability = str(self.get_parameter('reliability').value)
        if self._requested_fps <= 0.0:
            raise ValueError('fps must be positive')

        self._capture = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = cv2.VideoCapture(self._device)
        if not self._capture.isOpened():
            raise RuntimeError(
                f'Could not open camera {self._device}. Check the device path '
                'and camera permissions.'
            )

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
        self._capture.set(cv2.CAP_PROP_FPS, self._requested_fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._bridge = CvBridge()
        self._publisher = self.create_publisher(
            Image, self._topic, camera_qos(self._reliability)
        )
        self._timer = self.create_timer(
            1.0 / self._requested_fps, self._publish_frame
        )
        self._last_failure_log = -float('inf')
        self._frames_published = 0
        self._started = time.monotonic()

        actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._capture.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(
            f'Publishing {self._device} to {self._topic}: '
            f'{actual_width}x{actual_height} at camera-reported '
            f'{actual_fps:.1f} FPS; publisher limit {self._requested_fps:.1f} FPS; '
            f'QoS={self._reliability}'
        )

    def _publish_frame(self) -> None:
        success, frame = self._capture.read()
        if not success or frame is None:
            now = time.monotonic()
            if now - self._last_failure_log >= 2.0:
                self.get_logger().error(
                    f'Failed to read a frame from {self._device}'
                )
                self._last_failure_log = now
            return

        message = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._frame_id
        self._publisher.publish(message)
        self._frames_published += 1

        elapsed = time.monotonic() - self._started
        if self._frames_published % max(1, int(self._requested_fps * 5)) == 0:
            measured_fps = self._frames_published / max(elapsed, 1e-6)
            self.get_logger().info(
                f'Published {self._frames_published} frames '
                f'({measured_fps:.1f} FPS average)'
            )

    def destroy_node(self):
        """Release the physical camera before destroying the ROS node."""
        if self._capture is not None:
            self._capture.release()
        return super().destroy_node()


def main(args=None) -> None:
    """Run the temporary laptop-camera publisher."""
    rclpy.init(args=args)
    node = None
    try:
        node = LaptopCameraPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
