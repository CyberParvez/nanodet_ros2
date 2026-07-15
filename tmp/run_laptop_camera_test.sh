#!/usr/bin/env bash
set -eo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${1:-/dev/video0}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-15.0}"
DETECTOR_FPS="${DETECTOR_FPS:-5.0}"
ROS_RELIABILITY="${ROS_RELIABILITY:-reliable}"

if [[ "${CAMERA_FPS}" != *.* ]]; then
  CAMERA_FPS="${CAMERA_FPS}.0"
fi

if [[ ! -x "${WORKSPACE}/.venv/bin/python" ]]; then
  echo "Missing ${WORKSPACE}/.venv. Follow the build steps in README.md first." >&2
  exit 1
fi

if [[ ! -f "${WORKSPACE}/install/setup.bash" ]]; then
  echo "The ROS workspace is not built. Follow the build steps in README.md first." >&2
  exit 1
fi

source /opt/ros/jazzy/setup.bash
source "${WORKSPACE}/install/setup.bash"
set -u

mkdir -p "${WORKSPACE}/tmp/ros_logs"
export ROS_LOG_DIR="${WORKSPACE}/tmp/ros_logs"

detector_pid=""
cleanup() {
  if [[ -n "${detector_pid}" ]] && kill -0 "${detector_pid}" 2>/dev/null; then
    kill -INT "${detector_pid}" 2>/dev/null || true
    wait "${detector_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ros2 launch nanodet_ros2 detector.launch.py \
  input_topic:=/camera/image_raw \
  max_rate_hz:="${DETECTOR_FPS}" \
  input_reliability:="${ROS_RELIABILITY}" \
  output_reliability:="${ROS_RELIABILITY}" &
detector_pid=$!

"${WORKSPACE}/.venv/bin/python" \
  "${WORKSPACE}/tmp/laptop_camera_publisher.py" \
  --ros-args \
  -p device:="${DEVICE}" \
  -p topic:=/camera/image_raw \
  -p width:="${CAMERA_WIDTH}" \
  -p height:="${CAMERA_HEIGHT}" \
  -p fps:="${CAMERA_FPS}" \
  -p reliability:="${ROS_RELIABILITY}"
