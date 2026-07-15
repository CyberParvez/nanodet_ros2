# NanoDet ROS 2 camera detection

This workspace contains a resource-conscious ROS 2 Jazzy node that runs
NanoDet-Plus-m at 320 x 320 on a live image topic. It publishes both standard
2D detection messages and an optional image with bounding boxes.

The repository, workspace, and ROS package are named `nanodet_ros2`. This name
reflects the current bounding-box detector without implying that image
segmentation is implemented.

## Published interface

| Direction | Default topic | Type |
| --- | --- | --- |
| Subscribed | `/camera/image_raw` | `sensor_msgs/msg/Image` |
| Published | `/nanodet/detections` | `vision_msgs/msg/Detection2DArray` |
| Published | `/nanodet/image` | `sensor_msgs/msg/Image` |

The output image keeps the input image's timestamp and frame ID. The detection
message contains COCO class names, confidence scores, and boxes in the original
camera-image coordinates.

## Build

```bash
cd ~/nanodet_ros2
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m colcon build --symlink-install
source install/setup.bash
```

The workspace-local virtual environment supplies ONNX Runtime while retaining
access to the ROS packages installed under `/opt/ros/jazzy`. Building through
`.venv/bin/python` also makes the installed detector executable use that
environment automatically.

## Run

Start your camera driver first, then run:

```bash
ros2 launch nanodet_ros2 detector.launch.py
```

For a differently named image topic:

```bash
ros2 launch nanodet_ros2 detector.launch.py \
  input_topic:=/camera/color/image_raw \
  max_rate_hz:=5.0
```

View the annotated output with a ROS image viewer:

```bash
rqt_image_view /nanodet/image
```

Inspect structured detections:

```bash
ros2 topic echo /nanodet/detections
```

## Temporary laptop-camera test

A disposable webcam publisher and combined test command are available under
`tmp/`:

```bash
./tmp/run_laptop_camera_test.sh
```

It publishes the laptop camera on `/camera/image_raw`, starts NanoDet, and
publishes the annotated result on `/nanodet/image`. Pass another V4L2 path when
the webcam is not `/dev/video0`:

```bash
./tmp/run_laptop_camera_test.sh /dev/video2
```

See `tmp/README.md` for the complete temporary-test workflow.

## Resource controls

The defaults are intended for visualization rather than robot control:

- Inference is limited to 5 Hz even if the camera publishes faster.
- Rate-limited inference runs from a timer and always consumes the newest frame,
  avoiding frame-rate aliasing when camera frames arrive near the target rate.
- The subscription queue retains only the newest image.
- Inference uses ONNX Runtime on the CPU, without PyTorch. OpenCV DNN remains
  available as a slower fallback when ONNX Runtime is unavailable.
- ONNX Runtime worker spinning is disabled by default so idle inference threads
  sleep instead of consuming CPU between frames. Set
  `runtime_allow_spinning:=true` when comparing latency-oriented behavior.
- Bounding boxes are drawn only while the annotated-image topic has a
  subscriber.
- Set `publish_annotated:=false` to publish detection data only.

All parameters can be edited in
`src/nanodet_ros2/config/detector.yaml` or overridden on launch. Useful launch
arguments include `confidence_threshold`, `nms_threshold`, `max_rate_hz`,
`runtime`, `runtime_threads`, `runtime_allow_spinning`, `publish_annotated`, and
`model_path`. `input_reliability` and `output_reliability` accept `best_effort`
or `reliable`; the camera publisher and detector input must use compatible QoS.

### Watch live detection and CPU usage

Run the temporary camera publisher in one terminal:

```bash
cd ~/nanodet_ros2
source /opt/ros/jazzy/setup.bash
source install/setup.bash
.venv/bin/python tmp/laptop_camera_publisher.py --ros-args \
  -p device:=/dev/video0 \
  -p topic:=/camera/image_raw \
  -p width:=640 -p height:=480 -p fps:=6.0 \
  -p reliability:=reliable
```

Run the resource-limited detector in a second terminal:

```bash
cd ~/nanodet_ros2
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch nanodet_ros2 detector.launch.py \
  runtime:=onnxruntime runtime_threads:=2 \
  runtime_allow_spinning:=false max_rate_hz:=5.0 \
  input_reliability:=reliable output_reliability:=reliable
```

Open `htop` in a third terminal and filter for `detector_node` with `F4`. Open
RViz in another terminal, add an **Image** display, select `/nanodet/image`, and
set reliability to **Reliable**. To compare CPU usage, stop only the detector
and rerun the same launch command with `runtime_allow_spinning:=true`; leave the
camera and RViz running so the workload stays comparable.

## Model scope

The bundled model is trained on the 80 COCO categories. It recognizes common
objects such as people, cars, bicycles, chairs, and bottles. Detecting custom
robot-workspace objects requires a separately trained NanoDet model with a
matching class list.

This output is suitable for visualization and experimentation. Do not use a
2D detector as the robot's only collision-avoidance or safety input.

## Demo

https://github.com/user-attachments/assets/a38bb6cf-d3de-47c0-9ebf-ba7ceb82b930

The recording shows NanoDet running on the live ROS 2 camera stream with a
person bounding box and inference-time overlay.

## Model attribution

`nanodet-plus-m_320.onnx` comes from the Apache-2.0 licensed
[RangiLyu/nanodet](https://github.com/RangiLyu/nanodet) v1.0.0-alpha-1 release.
The model expects BGR input normalized with the official NanoDet configuration.
