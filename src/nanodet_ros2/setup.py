from glob import glob
from setuptools import find_packages, setup


package_name = "nanodet_ros2"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/models", glob("models/*")),
    ],
    install_requires=["setuptools", "onnxruntime>=1.19,<2"],
    zip_safe=True,
    maintainer="parvez",
    maintainer_email="parvez@todo.todo",
    description="Resource-conscious NanoDet object detection for ROS 2 images.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "detector_node = nanodet_ros2.detector_node:main",
        ],
    },
)
