#!/usr/bin/env python3
"""
Transformation utility functions for converting between different coordinate frames.
Provides functions to work with quaternions, transformation matrices and ROS pose messages.
"""
from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray, PoseStamped
from geometry_msgs.msg import PointStamped, TransformStamped
from std_msgs.msg import Header
import numpy
# import rospy  # Commented out, might be needed for ROS1
from scipy.spatial.transform import Rotation as Rot
import math

# Small value to prevent numerical issues in quaternion calculations
_EPS = numpy.finfo(float).eps * 4.0


def quaternion_matrix(quaternion):
    """
    Return homogeneous rotation matrix from quaternion.
    
    Args:
        quaternion: list or array with quaternion coefficients (x, y, z, w)
        
    Returns:
        4x4 homogeneous rotation matrix
        
    Examples:
        >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
        >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
        True
    """
    q = numpy.array(quaternion[:4], dtype=numpy.float64, copy=True)
    nq = numpy.dot(q, q)
    if nq < _EPS:
        return numpy.identity(4)
    
    q *= math.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    
    return numpy.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
    ), dtype=numpy.float64)


def quaternion_from_matrix(matrix):
    """
    Return quaternion from rotation matrix.
    
    Args:
        matrix: 4x4 rotation matrix or 3x3 rotation matrix
        
    Returns:
        Quaternion as [x, y, z, w]
        
    Examples:
        >>> R = rotation_matrix(0.123, (1, 2, 3))
        >>> q = quaternion_from_matrix(R)
        >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
        True
    """
    q = numpy.empty((4, ), dtype=numpy.float64)
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    t = numpy.trace(M)
    
    if t > M[3, 3]:
        # Standard case
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        # Handle degenerate case
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


def pose_stamped_to_mat(p):
    """
    Convert a PoseStamped message to a 4x4 transformation matrix.
    
    Args:
        p: PoseStamped message
        
    Returns:
        4x4 transformation matrix
    """
    q = p.pose.orientation
    pos = p.pose.position
    
    # Create rotation matrix from quaternion
    T = quaternion_matrix([q.x, q.y, q.z, q.w])
    
    # Add translation component
    T[:3, 3] = numpy.array([pos.x, pos.y, pos.z])
    
    return T


def pose_to_mat(p):
    """
    Convert a Pose message to a 4x4 transformation matrix.
    
    Args:
        p: Pose message
        
    Returns:
        4x4 transformation matrix
    """
    q = p.orientation
    pos = p.position
    
    # Create rotation matrix from quaternion
    T = quaternion_matrix([q.x, q.y, q.z, q.w])
    
    # Add translation component
    T[:3, 3] = numpy.array([pos.x, pos.y, pos.z])
    
    return T


def mat_to_pose_stamped(m, frame_id="test"):
    """
    Convert a transformation matrix to a PoseStamped message.
    
    Args:
        m: 4x4 transformation matrix
        frame_id: frame ID for the header
        
    Returns:
        PoseStamped message
    """
    q = quaternion_from_matrix(m)
    p = PoseStamped(
        header=Header(frame_id=frame_id),  # robot.get_planning_frame()
        pose=Pose(
            position=Point(*m[:3, 3]),
            orientation=Quaternion(*q)
        )
    )
    return p


def transform_inverse(transform_in):
    """
    Compute the inverse of a transformation matrix.
    
    Args:
        transform_in: 4x4 transformation matrix
        
    Returns:
        4x4 inverse transformation matrix
    """
    # Extract rotation matrix and translation vector
    rot_in = transform_in[:3, :3]
    trans_in = transform_in[:3, [-1]]
    
    # Compute inverse rotation (transpose) and inverse translation
    rot_out = rot_in.T
    trans_out = -numpy.matmul(rot_out, trans_in)
    
    # Construct the inverse transformation matrix
    return numpy.vstack((
        numpy.hstack((rot_out, trans_out)),
        numpy.array([0, 0, 0, 1])
    ))


def transformation(pose):
    """
    Transform a pose from camera frame to base frame.
    
    Args:
        pose: list containing position and orientation [x, y, z, qx, qy, qz, qw]
        
    Returns:
        list containing transformed pose [x, y, z, qw, qx, qy, qz]
    """
    # Object with respect to Camera - create Pose message
    p_obj_wrt_cam = Pose()
    p_obj_wrt_cam.position.x = float(pose[0])
    p_obj_wrt_cam.position.y = float(pose[1])
    p_obj_wrt_cam.position.z = float(pose[2])
    p_obj_wrt_cam.orientation.w = float(pose[6])
    p_obj_wrt_cam.orientation.x = float(pose[3])
    p_obj_wrt_cam.orientation.y = float(pose[4])
    p_obj_wrt_cam.orientation.z = float(pose[5])
    
    # Convert to transformation matrix (camera to object)
    T_cam_obj = pose_to_mat(p_obj_wrt_cam)

    # Camera with respect to Base - static transformation
    p_cam_wrt_base = Pose()
    p_cam_wrt_base.position.x = -0.143361
    p_cam_wrt_base.position.y = -1.45842
    p_cam_wrt_base.position.z = 0.375607
    p_cam_wrt_base.orientation.w = 0.575573
    p_cam_wrt_base.orientation.x = -0.817741
    p_cam_wrt_base.orientation.y = -0.00388839
    p_cam_wrt_base.orientation.z = -0.000290818
    
    # Convert to transformation matrix (base to camera)
    T_base_cam = pose_to_mat(p_cam_wrt_base)

    # Calculate object with respect to Base by matrix multiplication
    T_base_obj = numpy.matmul(T_base_cam, T_cam_obj)

    # Extract rotation and position
    rot_mat = Rot.from_matrix(T_base_obj[:3, :3])
    quat = rot_mat.as_quat()  # [x, y, z, w] format
    position = T_base_obj[:3, 3]  # [x, y, z]

    # Combine position and quaternion
    pose_combined = numpy.concatenate((position, quat))
    
    # Reorder to [x, y, z, qw, qx, qy, qz] format
    object_pose_base = [
        pose_combined[0],  # x
        pose_combined[1],  # y
        pose_combined[2],  # z
        pose_combined[6],  # qw
        pose_combined[3],  # qx
        pose_combined[4],  # qy
        pose_combined[5]   # qz
    ]
    
    return object_pose_base
