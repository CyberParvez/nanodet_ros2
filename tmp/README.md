# Temporary laptop-camera test

This folder contains development utilities for testing the `nanodet_ros2`
package with a disposable webcam publisher. Generated ROS logs and Python cache
files remain ignored by Git.

Run from the workspace root:

```bash
./tmp/run_laptop_camera_test.sh
```

The first argument selects a different V4L2 device if `/dev/video0` is not the
laptop camera:

```bash
./tmp/run_laptop_camera_test.sh /dev/video2
```

The script starts:

- a webcam publisher on `/camera/image_raw` at up to 15 FPS;
- the NanoDet detector at up to 5 FPS;
- structured detections on `/nanodet/detections`;
- annotated frames on `/nanodet/image` when a viewer subscribes.

The temporary harness uses Reliable QoS to avoid dropping large raw image
samples on the local DDS transport. For a lower-bandwidth 5 FPS visualization
test, override the camera settings without editing the script:

```bash
CAMERA_WIDTH=320 CAMERA_HEIGHT=240 CAMERA_FPS=6 DETECTOR_FPS=5 \
  ./tmp/run_laptop_camera_test.sh
```

In another terminal, view the annotated image:

```bash
cd ~/nanodet_ros2
source /opt/ros/jazzy/setup.bash
source install/setup.bash
rqt_image_view /nanodet/image
```

Stop the test with `Ctrl+C`. The wrapper also stops the detector process and
releases the webcam.

To find the camera device on the host:

```bash
ls -l /dev/video*
```
