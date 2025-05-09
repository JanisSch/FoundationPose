# FoundationPoseROS2: Multi-Object Pose Estimation and Tracking in ROS2

## Description

FoundationPoseROS2 is a ROS2-based implementation of FoundationPose for 6D object pose estimation and tracking. The system enables real-time detection, pose estimation, and tracking of multiple objects simultaneously using a RealSense camera and the Segment Anything Model 2 (SAM2).

### Key Features:
- Multi-object pose estimation and tracking
- SAM2-based automatic segmentation
- ROS2 integration for use with robotic systems
- Camera-to-base frame transformation
- Simple GUI for object selection
- Works with standard 8GB NVIDIA GPUs

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Starting the Camera](#starting-the-camera)
  - [Running FoundationPose](#running-foundationpose)
  - [Using the Launch File](#using-the-launch-file-recommended)
  - [Visualization with RViz](#visualization-with-rviz)
  - [Reset Object Detection](#reset-object-detection)
- [Adding New Objects](#adding-new-objects)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

## Installation

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

### Fix Library Issues

If you encounter issues with libstdc++, use the following commands to fix them:

```bash
# Replace incompatible libstdc++ with system version
mv /home/jscheidegger/anaconda3/envs/found/lib/libstdc++.so.6 /home/jscheidegger/anaconda3/envs/found/lib/libstdc++.so.6.bak
ln -s /usr/lib/gcc/x86_64-linux-gnu/13/../../../x86_64-linux-gnu/libstdc++.so.6 /home/jscheidegger/anaconda3/envs/found/lib/libstdc++.so.6
```

### Fix Permission Issues

If you encounter permission issues with Python modules:

```bash
# Make Python modules executable
sudo chmod 755 /path/to/module.cpython-312-x86_64-linux-gnu.so
```

## Usage

### Starting the Camera

Start the RealSense camera with:

```bash
# Basic camera launch
source /opt/ros/jazzy/setup.bash && ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true pointcloud.enable:=true
```

Or with custom camera parameters:

```bash
# With custom camera parameters
source /opt/ros/jazzy/setup.bash && ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true pointcloud.enable:=true params_file:=/home/jscheidegger/Documents/New_Try/static_camera_info.yaml
```

### Running FoundationPose

Run the FoundationPose node manually:

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

This launch file automatically starts the camera node and the FoundationPose node, making the setup process much simpler.

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

To track new objects:

1. Add mesh files (.obj or .stl) to:
   ```
   ./FoundationPoseROS2/demo_data/object_name/<OBJECT_MESH>.obj
   ```

2. When running the application, a GUI will appear allowing you to select which objects to track and specify the maximum number of instances of each object that might appear in the scene.

## Configuration

### Camera Configuration

The system uses the camera parameters from the RealSense camera. Custom camera parameters can be specified using a YAML file passed to the RealSense launch file.

Example camera configuration file (`static_camera_info.yaml`):
```yaml
camera_matrix:
  data: [614.787, 0, 324.18, 0, 614.622, 237.78, 0, 0, 1]
distortion_coefficients:
  data: [0, 0, 0, 0, 0]
```

## Project Structure

```
/home/jscheidegger/Documents/New_Try/
├── FoundationPoseROS2/
│   ├── FoundationPose/          # Core pose estimation algorithms
│   ├── demo_data/               # Example mesh files
│   ├── foundationpose_ros_multi.py # Main ROS2 node
│   ├── cam_2_base_transform.py  # Transformation utilities
│   ├── foundationpose_launch.py # Launch file
│   └── foundationpose_ros.rviz  # RViz configuration
├── static_camera_info.yaml      # Camera calibration parameters
└── README.md                    # This documentation
```

## Troubleshooting

### Common Issues:

1. **Library Errors**: If you encounter errors related to CUDA or libraries, ensure you've set the correct CUDA path:
   ```bash
   export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
   ```

2. **No Objects Detected**: Try publishing to the `/redo` topic to force re-detection.

3. **Camera Not Found**: Ensure the RealSense camera is properly connected and the camera node is running.

4. **Permission Issues**: If Python modules are not executable, use the chmod command mentioned in the installation section.

5. **Different Camera Topics**: If using a different camera, modify the topic names in `foundationpose_ros_multi.py`.

## License

This project is released under the MIT License.

## Contact

For questions or issues, please open an issue on GitHub or contact the project maintainer.
