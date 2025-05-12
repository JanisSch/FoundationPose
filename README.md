# Multi-Object Pose Estimation and Tracking in ROS2

## Description

FoundationPose is a ROS2-based implementation for multi-object pose estimation and tracking. The system leverages NVIDIA FoundationPose for pose estimation, SAM (Segment Anything Model) for automatic segmentation, and SIFT (Scale-Invariant Feature Transform) for feature matching during the initial detection. It enables real-time detection, pose estimation, and tracking of multiple objects, seamlessly integrating with robotic systems.

### Key Features:
- Multi-object pose estimation and tracking
- ROS2 integration for use with robotic systems
- Camera-to-base frame transformation
- Simple GUI for object selection
- Works with standard 8GB NVIDIA GPUs

## Table of Contents
- [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Dependencies](#dependencies)
   - [Conda Environment Setup](#conda-environment-setup)
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

## Installation

### Prerequisites
- Ubuntu
- ROS2 Jazzy Jalisco
- Minimum 8GB NVIDIA GPU
- CUDA 12.x
- Intel RealSense camera
- Python 3.10 or later

### Dependencies

```bash
# Install ROS2 on Ubuntu
sudo apt install ros-<ROS_DISTRO>-desktop

# Install librealsense2
sudo apt install ros-<ROS_DISTRO>-librealsense2*

# Install debian realsense2 package
sudo apt install ros-<ROS_DISTRO>-realsense2-*

# Setup CUDA 12.x
sudo apt-get --purge remove 'nvidia-*'
sudo apt-get autoremove
sudo reboot

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# Install Miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
```

### Conda Environment Setup
```bash
# Clone repository
git clone https://github.com/JanisSch/FoundationPose.git
```

```bash
# Create conda environment
conda create -n found python=3.10 -y

# Activate conda environment
conda activate found
```
Run the following commands to build the extensions:

```bash
cd FoundationPoseROS2 && export PATH=/usr/local/<YOUR_cuda-12.X_VERSION>/bin${PATH:+:${PATH}}~ && bash build_all_conda.sh
```

**Important:**
In the setup.py file located at /FoundationPose/bundlesdf/mycuda/, the C++ flags should be updated from C++14 to C++17 for compatibility with newer Nvidia GPUs.
This can be done by modifying lines 18 and 19 in the file's nvcc_flags and c_flags sections.

---
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
source /opt/ros/jazzy/setup.bash && ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true pointcloud.enable:=true params_file:path_to_your_file/static_camera_info.yaml
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
ros2 launch path_to_your_file/foundationpose_launch.py
```

This launch file automatically starts the camera node and the FoundationPose node, making the setup process much simpler.

### Visualization with RViz

To visualize the results in RViz using a pre-configured setup:

```bash
rviz2 path_to_your_file/foundationpose_ros.rviz
```
Note: If you start the application using foundationpose_launch.py, RViz will be launched automatically with the pre-configured setup.

### Reset Object Detection

If you need to force the system to re-detect objects:

```bash
ros2 topic pub --once /redo std_msgs/msg/Bool '{data: true}'
```

Note: This topic can also be used by a robot performing automatic robotic assembly to publish a message, prompting the system to detect new objects in the environment.

## Adding New Objects

To track new objects:

1. Add mesh files (.obj or .stl) to:
   ```bash
   ./FoundationPoseROS2/demo_data/object_name/<OBJECT_MESH>.obj
   ```
2. Run render_silhouettes.py to generate binary masks for the object. These masks will be saved in the following directory:
   ```
   ./FoundationPoseROS2/demo_data/object_name/object_name_silhouettes
   ```
3. When running the application, a GUI will appear allowing you to select which objects to track and specify the maximum number of instances of each object that might appear in the scene.

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

4. **Different Camera Topics**: If using a different camera, modify the topic names in `foundationpose_ros_multi.py`.

## License

This project is released under the MIT License.
