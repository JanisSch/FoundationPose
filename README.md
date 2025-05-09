# FoundationPoseROS2: Multi-Object Pose Estimation and Tracking in ROS2

This repository contains a ROS2-based implementation of FoundationPose for 6D object pose estimation and tracking. The system uses a RealSense camera and the Segment Anything Model 2 (SAM2) for real-time pose estimation of objects.

## Setup

### Dependencies

- Ubuntu
- ROS2 Jazzy Jalisco
- NVIDIA GPU with CUDA 12.4
- Intel RealSense camera
- Python 3.10 or later

### Conda Environment Setup

Create and activate a Conda environment:

```bash
# Create conda environment
conda create -n found python=3.10 -y

# Activate conda environment
conda activate found
```

### Library Issues

If you encounter issues with libstdc++, use the following commands to fix them:

```bash
# Replace incompatible libstdc++ with system version
mv /home/jscheidegger/anaconda3/envs/found/lib/libstdc++.so.6 /home/jscheidegger/anaconda3/envs/found/lib/libstdc++.so.6.bak
ln -s /usr/lib/gcc/x86_64-linux-gnu/13/../../../x86_64-linux-gnu/libstdc++.so.6 /home/jscheidegger/anaconda3/envs/found/lib/libstdc++.so.6
```

### Permissions

If you encounter permission issues with Python modules:

```bash
# Make Python modules executable
sudo chmod 755 /path/to/module.cpython-312-x86_64-linux-gnu.so
```

## Running the Application

### Starting the RealSense Camera

```bash
# Basic camera launch
source /opt/ros/jazzy/setup.bash && ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true pointcloud.enable:=true

# With custom camera parameters
source /opt/ros/jazzy/setup.bash && ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true pointcloud.enable:=true params_file:=/home/jscheidegger/Documents/New_Try/static_camera_info.yaml
```

### Running FoundationPose Manually

```bash
# Activate conda environment first
conda activate found

# Run FoundationPose with CUDA
source /opt/ros/jazzy/setup.bash && export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}} && python ./FoundationPoseROS2/foundationpose_ros_multi.py
```

### Using the Launch File (Recommended)

The easiest way to run the application is using the provided launch file which sets up all necessary components:

```bash
# Run everything with a single launch file
ros2 launch /home/jscheidegger/Documents/New_Try/FoundationPoseROS2/foundationpose_launch.py
```

### Visualization with RViz

To visualize the results in RViz using a pre-configured setup:

```bash
rviz2 -d /home/jscheidegger/Documents/New_Try/FoundationPoseROS2/foundationpose_ros.rviz
```

### Reset Object Detection

If you need to force the system to re-detect objects:

```bash
ros2 topic pub --once /redo std_msgs/msg/Bool '{data: true}'
```

## Adding New Objects

To track new objects, add mesh files (.obj or .stl) to:

```
./FoundationPoseROS2/demo_data/object_name/<OBJECT_MESH>.obj
```

When running the application, a GUI will appear allowing you to select which objects to track.

## Features

- Multi-object pose estimation and tracking
- SAM2-based automatic segmentation
- ROS2 integration for use with robotic systems
- Camera-to-base frame transformation
- Simple GUI for object selection
- Visualization of object poses with axes and bounding boxes

## Topics

The system publishes:
- TF frames for each detected object
- Individual pose topics for each object: `/<object_name>_<instance>_position`
- Listen to `/redo` topic to trigger re-detection

## Camera Configuration

The system uses the camera parameters from the RealSense camera. Custom camera parameters can be specified using a YAML file passed to the RealSense launch file.
