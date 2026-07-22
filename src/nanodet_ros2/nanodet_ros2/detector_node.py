"""ROS 2 node that applies NanoDet to a sensor_msgs/Image stream."""

import os
import re
import time

import cv2
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .nanodet_engine import Detection, NanoDet


def sensor_qos(
    depth: int = 1, reliability: str = "best_effort"
) -> QoSProfile:
    """QoS that favors current camera frames over queued stale frames."""

    reliability_policies = {
        "best_effort": ReliabilityPolicy.BEST_EFFORT,
        "reliable": ReliabilityPolicy.RELIABLE,
    }
    normalized_reliability = reliability.strip().lower()
    if normalized_reliability not in reliability_policies:
        raise ValueError("reliability must be best_effort or reliable")

    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=reliability_policies[normalized_reliability],
        durability=DurabilityPolicy.VOLATILE,
    )


class NanoDetNode(Node):
    """Subscribe to camera images and publish NanoDet results."""

    def __init__(self) -> None:
        super().__init__("nanodet_detector")

        package_share = get_package_share_directory("nanodet_ros2")
        default_model = os.path.join(
            package_share, "models", "nanodet-plus-m_320.onnx"
        )

        self.declare_parameter("model_path", "")
        self.declare_parameter("input_topic", "/camera/image_raw")
        self.declare_parameter("detections_topic", "/nanodet/detections")
        self.declare_parameter("annotated_topic", "/nanodet/image")
        self.declare_parameter("confidence_threshold", 0.4)
        self.declare_parameter("nms_threshold", 0.6)
        self.declare_parameter("max_detections", 100)
        self.declare_parameter("max_rate_hz", 10.0)
        self.declare_parameter("input_reliability", "best_effort")
        self.declare_parameter("output_reliability", "best_effort")
        self.declare_parameter("allowed_labels", "")
        self.declare_parameter("runtime", "auto")
        self.declare_parameter("runtime_threads", 4)
        self.declare_parameter("runtime_allow_spinning", False)
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("draw_performance", True)

        model_path = str(self.get_parameter("model_path").value).strip()
        if not model_path:
            model_path = default_model
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"NanoDet model not found: {model_path}")

        self._max_rate_hz = float(self.get_parameter("max_rate_hz").value)
        if self._max_rate_hz < 0.0:
            raise ValueError("max_rate_hz must be non-negative")
        self._minimum_period = (
            1.0 / self._max_rate_hz if self._max_rate_hz > 0.0 else 0.0
        )
        self._latest_image_message = None
        self._publish_annotated = bool(
            self.get_parameter("publish_annotated").value
        )
        self._draw_performance = bool(
            self.get_parameter("draw_performance").value
        )
        self._allowed_labels = self._parse_allowed_labels(
            str(self.get_parameter("allowed_labels").value)
        )
        self._last_status_log = -float("inf")
        self._processed_since_status = 0
        self._annotated_since_status = 0

        self._bridge = CvBridge()
        self._detector = NanoDet(
            model_path=model_path,
            score_threshold=float(
                self.get_parameter("confidence_threshold").value
            ),
            nms_threshold=float(self.get_parameter("nms_threshold").value),
            max_detections=int(self.get_parameter("max_detections").value),
            runtime=str(self.get_parameter("runtime").value),
            runtime_threads=int(self.get_parameter("runtime_threads").value),
            runtime_allow_spinning=bool(
                self.get_parameter("runtime_allow_spinning").value
            ),
        )

        input_reliability = str(self.get_parameter("input_reliability").value)
        output_reliability = str(self.get_parameter("output_reliability").value)
        input_qos = sensor_qos(reliability=input_reliability)
        output_qos = sensor_qos(reliability=output_reliability)
        detections_topic = str(self.get_parameter("detections_topic").value)
        annotated_topic = str(self.get_parameter("annotated_topic").value)
        input_topic = str(self.get_parameter("input_topic").value)
        self._detections_publisher = self.create_publisher(
            Detection2DArray, detections_topic, output_qos
        )
        self._annotated_publisher = self.create_publisher(
            Image, annotated_topic, output_qos
        )
        if self._minimum_period > 0.0:
            self._subscription = self.create_subscription(
                Image, input_topic, self._store_latest_image, input_qos
            )
            self._inference_timer = self.create_timer(
                self._minimum_period, self._process_latest_image
            )
        else:
            self._subscription = self.create_subscription(
                Image, input_topic, self._process_image, input_qos
            )
            self._inference_timer = None

        self.get_logger().info(
            f"NanoDet ready: {input_topic} -> {detections_topic}, {annotated_topic}; "
            f"rate limit={self._max_rate_hz:.1f} Hz; "
            f"runtime={self._detector.runtime_name}; "
            f"runtime spinning="
            f"{self.get_parameter('runtime_allow_spinning').value}; "
            f"QoS={input_reliability} input/{output_reliability} output; "
            f"model={model_path}"
            f"{'; labels=' + ','.join(sorted(self._allowed_labels)) if self._allowed_labels else ''}"
        )

    def _store_latest_image(self, message: Image) -> None:
        """Retain the newest frame without coupling input and inference rates."""
        self._latest_image_message = message

    def _process_latest_image(self) -> None:
        """Process at the configured rate using the freshest available frame."""
        message = self._latest_image_message
        if message is None:
            return
        self._latest_image_message = None
        self._process_image(message)

    def _process_image(self, message: Image) -> None:
        now = time.monotonic()

        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            result = self._detector.detect(image)
            detections = self._filter_detections(result.detections)
            self._processed_since_status += 1
        except Exception as error:  # Keep a malformed frame from stopping the node.
            self.get_logger().error(f"Image inference failed: {error}")
            return

        detection_message = self._to_detection_message(message, detections)
        if not self._publish_if_active(
            self._detections_publisher, detection_message
        ):
            return

        has_image_subscriber = self._annotated_publisher.get_subscription_count() > 0
        if self._publish_annotated and has_image_subscriber:
            annotated = self._draw_detections(image, detections, result.inference_ms)
            annotated_message = self._bridge.cv2_to_imgmsg(
                annotated, encoding="bgr8"
            )
            annotated_message.header = message.header
            if not self._publish_if_active(
                self._annotated_publisher, annotated_message
            ):
                return
            self._annotated_since_status += 1

        if now - self._last_status_log >= 5.0:
            status_period = (
                now - self._last_status_log
                if self._last_status_log > 0.0
                else 0.0
            )
            rates = ""
            if status_period > 0.0:
                rates = (
                    f"; processing={self._processed_since_status / status_period:.1f} Hz"
                    f"; annotated={self._annotated_since_status / status_period:.1f} Hz"
                )
            self.get_logger().info(
                f"Detected {len(detections)} objects in "
                f"{result.inference_ms:.1f} ms{rates}"
            )
            self._last_status_log = now
            self._processed_since_status = 0
            self._annotated_since_status = 0

    @staticmethod
    def _parse_allowed_labels(value: str) -> frozenset[str]:
        """Parse a comma- or whitespace-separated label allow-list."""
        labels = {
            part.strip().lower()
            for part in re.split(r"[,\s]+", value.strip())
            if part.strip()
        }
        return frozenset(labels)

    def _filter_detections(
        self, detections: tuple[Detection, ...]
    ) -> tuple[Detection, ...]:
        """Keep only detections whose labels are explicitly allowed."""
        if not self._allowed_labels:
            return detections
        return tuple(
            detection
            for detection in detections
            if detection.label.lower() in self._allowed_labels
        )

    @staticmethod
    def _publish_if_active(publisher, message) -> bool:
        """Publish unless ROS shutdown invalidates the context mid-callback."""
        if not rclpy.ok():
            return False
        try:
            publisher.publish(message)
        except Exception:
            if rclpy.ok():
                raise
            return False
        return True

    @staticmethod
    def _to_detection_message(
        image_message: Image, detections: tuple[Detection, ...]
    ) -> Detection2DArray:
        output = Detection2DArray()
        output.header = image_message.header

        for item in detections:
            x1, y1, x2, y2 = item.box
            detection = Detection2D()
            detection.header = image_message.header
            detection.bbox.center.position.x = (x1 + x2) / 2.0
            detection.bbox.center.position.y = (y1 + y2) / 2.0
            detection.bbox.size_x = max(0.0, x2 - x1)
            detection.bbox.size_y = max(0.0, y2 - y1)

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = item.label
            hypothesis.hypothesis.score = item.score
            detection.results.append(hypothesis)
            output.detections.append(detection)

        return output

    def _draw_detections(
        self,
        image,
        detections: tuple[Detection, ...],
        inference_ms: float,
    ):
        output = image.copy()
        for item in detections:
            x1, y1, x2, y2 = (int(round(value)) for value in item.box)
            color = self._class_color(item.class_id)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            text = f"{item.label} {item.score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            text_top = max(0, y1 - text_height - baseline - 4)
            cv2.rectangle(
                output,
                (x1, text_top),
                (x1 + text_width + 4, text_top + text_height + baseline + 4),
                color,
                -1,
            )
            cv2.putText(
                output,
                text,
                (x1 + 2, text_top + text_height + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        if self._draw_performance:
            text = f"NanoDet {inference_ms:.1f} ms | {len(detections)} objects"
            cv2.putText(
                output,
                text,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (40, 255, 40),
                2,
                cv2.LINE_AA,
            )
        return output

    @staticmethod
    def _class_color(class_id: int) -> tuple[int, int, int]:
        return (
            int((37 * class_id + 80) % 256),
            int((17 * class_id + 160) % 256),
            int((29 * class_id + 220) % 256),
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = NanoDetNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
