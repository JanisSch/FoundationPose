import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    """
    Generate launch description for FoundationPose ROS2 system.
    
    This launches:
    1. RealSense camera node with RGBD, sync, and alignment enabled
    2. FoundationPose multi-object detection and tracking node
    3. RViz2 for visualization
    
    Returns:
        LaunchDescription: The complete launch configuration
    """
    # Define the path to the project directory
    project_dir = '/home/jscheidegger/Documents/New_Try'  # CHANGE THIS: Base project directory path
    
    # Define path to the camera parameters file
    camera_params_file = os.path.join(project_dir, 'static_camera_info.yaml')
    
    # Define path to the RViz configuration
    rviz_config = os.path.join(project_dir, 'FoundationPoseROS2', 'foundationpose_ros.rviz') 
    # Define the foundationpose script path
    foundation_pose_script = os.path.join(
        project_dir, 
        'FoundationPoseROS2', 
        'foundationpose_ros_multi.py'
    )  # CHANGE THIS: Path to your FoundationPose script
    
    # Get the RealSense launch file from its package directory
    realsense_launch_dir = get_package_share_directory('realsense2_camera')
    realsense_launch_path = os.path.join(realsense_launch_dir, 'launch')
    
    # Launch RealSense camera using the launch file
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([realsense_launch_path, '/rs_launch.py']),
        launch_arguments={
            'enable_rgbd': 'true',
            'enable_sync': 'true',
            'align_depth.enable': 'true', 
            'enable_color': 'true',
            'enable_depth': 'true',
            'pointcloud.enable': 'true',
            'params_file': camera_params_file
        }.items()
    )
    
    # Launch FoundationPose node as an ExecuteProcess
    foundation_pose_node = ExecuteProcess(
        cmd=[
            'bash', '-c', 
            f'source /opt/ros/jazzy/setup.bash && '  # CHANGE THIS: Path to your ROS distribution
            f'export PATH=/usr/local/cuda-12.4/bin${{PATH:+:${{PATH}}}} && '  # CHANGE THIS: Path to your CUDA installation
            f'python {foundation_pose_script}'
        ],
        output='screen',
    )
    
    # Launch RViz2
    rviz_node = ExecuteProcess(
        cmd=['rviz2', '-d', rviz_config],
        output='screen',
    )
    
    # Create and return the launch description
    return LaunchDescription([
        realsense_launch,
        foundation_pose_node,
        rviz_node,
    ])
