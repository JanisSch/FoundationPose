import sys
sys.path.append('./FoundationPose')
sys.path.append('./FoundationPose/nvdiffrast')
sys.path.append('/home/jscheidegger/Documents/New_Try/FoundationPoseROS2')
sys.path.append('/home/jscheidegger/Documents/New_Try/FoundationPoseROS2/FoundationPose')

import rclpy
from rclpy.node import Node
from estimater import *
import cv2
import numpy as np
import trimesh
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped
from cv_bridge import CvBridge
import argparse
import os
from scipy.spatial.transform import Rotation as R
from ultralytics import SAM
from cam_2_base_transform import *
import os
import tkinter as tk
from tkinter import Listbox, END, Button
import glob
import logging  # Add logging for debugging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Save the original `__init__` and `register` methods
original_init = FoundationPose.__init__
original_register = FoundationPose.register

# Modify `__init__` to add `is_register` attribute
def modified_init(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer=None, refiner=None, glctx=None, debug=0, debug_dir='./FoundationPose'):
    original_init(self, model_pts, model_normals, symmetry_tfs, mesh, scorer, refiner, glctx, debug, debug_dir)
    self.is_register = False  # Initialize as False

# Modify `register` to set `is_register` to True when a pose is registered
def modified_register(self, K, rgb, depth, ob_mask, iteration):
    pose = original_register(self, K, rgb, depth, ob_mask, iteration)
    self.is_register = True  # Set to True after registration
    return pose

# Apply the modifications
FoundationPose.__init__ = modified_init
FoundationPose.register = modified_register

class FileSelectorGUI:
    def __init__(self, master, file_paths):
        self.master = master
        self.master.title("Library: Sequence Selector")
        self.file_paths = file_paths
        self.reordered_paths = None  # Store the reordered paths here

        # Create a listbox to display the file names
        self.listbox = Listbox(master, selectmode="extended", width=50, height=10)
        self.listbox.pack()

        # Populate the listbox with file names without extensions
        for file_path in self.file_paths:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            self.listbox.insert(END, file_name)

        # Buttons for rearranging the order
        self.up_button = Button(master, text="Move Up", command=self.move_up)
        self.up_button.pack(side="left", padx=5, pady=5)

        self.down_button = Button(master, text="Move Down", command=self.move_down)
        self.down_button.pack(side="left", padx=5, pady=5)

        self.done_button = Button(master, text="Done", command=self.done)
        self.done_button.pack(side="left", padx=5, pady=5)

    def move_up(self):
        """Move selected items up in the listbox."""
        selected_indices = list(self.listbox.curselection())
        for index in selected_indices:
            if index > 0:
                # Swap with the previous item
                file_name = self.listbox.get(index)
                self.listbox.delete(index)
                self.listbox.insert(index - 1, file_name)
                self.listbox.selection_set(index - 1)

    def move_down(self):
        """Move selected items down in the listbox."""
        selected_indices = list(self.listbox.curselection())
        for index in reversed(selected_indices):
            if index < self.listbox.size() - 1:
                # Swap with the next item
                file_name = self.listbox.get(index)
                self.listbox.delete(index)
                self.listbox.insert(index + 1, file_name)
                self.listbox.selection_set(index + 1)

    def done(self):
        """Save the reordered paths and close the GUI."""
        reordered_file_names = self.listbox.get(0, END)

        # Recreate the full file paths based on the reordered file names (without extensions)
        file_name_to_full_path = {
            os.path.splitext(os.path.basename(file))[0]: file for file in self.file_paths
        }
        self.reordered_paths = [file_name_to_full_path[file_name] for file_name in reordered_file_names]

        # Close the GUI
        self.master.quit()

    def get_reordered_paths(self):
        """Return the reordered file paths after the GUI has closed."""
        return self.reordered_paths

# Example usage
def rearrange_files(file_paths):
    root = tk.Tk()
    app = FileSelectorGUI(root, file_paths)
    root.mainloop()  # Start the GUI event loop
    reordered_paths = app.get_reordered_paths()  # Get reordered paths after GUI closes
    root.destroy()  # Ensure the GUI is properly closed
    return reordered_paths

# Argument Parser
parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument('--est_refine_iter', type=int, default=4)
parser.add_argument('--track_refine_iter', type=int, default=2)
args = parser.parse_args()

class PoseEstimationNode(Node):
    def __init__(self, new_file_paths):
        super().__init__('pose_estimation_node')
        
        # ROS subscriptions and publishers
        self.image_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        
        self.bridge = CvBridge()
        self.depth_image = None
        self.color_image = None
        self.cam_K = None  # Initialize cam_K as None until we receive the camera info
        
        # Load meshes
        self.mesh_files = new_file_paths
        self.meshes = [trimesh.load(mesh) for mesh in self.mesh_files]
        
        logging.info(f"Found 3D files: {self.meshes}")
        if not self.meshes:
            logging.error("No 3D files were loaded into self.meshes. Check file paths and processing logic.")
        
        self.bounds = [trimesh.bounds.oriented_bounds(mesh) for mesh in self.meshes]
        self.bboxes = [np.stack([-extents/2, extents/2], axis=0).reshape(2, 3) for _, extents in self.bounds]
        
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()

        # Initialize SAM2 model
        self.seg_model = SAM("sam2.1_b.pt")

        self.pose_estimations = {}  # Dictionary to track multiple pose estimations
        self.pose_publishers = {}  # Dictionary to store publishers for each object
        self.tracked_objects = []  # Initialize to store selected objects' masks
        self.i = 0
        self.initial_selection_done = False  # Track if initial selection is completed
        self.all_meshes = self.meshes.copy()
        self.all_bounds = self.bounds.copy()
        self.all_bboxes = self.bboxes.copy()
        self.initial_detection_done = False  # Flag to track if initial detection is completed

    def camera_info_callback(self, msg):
        if self.cam_K is None:  # Update cam_K only once to avoid redundant updates
            self.cam_K = np.array(msg.k).reshape((3, 3))
            self.get_logger().info(f"Camera intrinsic matrix initialized: {self.cam_K}")

    def image_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1") / 1e3
        self.process_images()

    def project_3d_bbox_to_2d(self, bbox_3d, cam_K):
        """Project a 3D bounding box into 2D using the camera intrinsic matrix."""
        points_3d = np.array([
            [bbox_3d[0][0], bbox_3d[0][1], bbox_3d[0][2]],
            [bbox_3d[1][0], bbox_3d[0][1], bbox_3d[0][2]],
            [bbox_3d[0][0], bbox_3d[1][1], bbox_3d[0][2]],
            [bbox_3d[1][0], bbox_3d[1][1], bbox_3d[0][2]],
            [bbox_3d[0][0], bbox_3d[0][1], bbox_3d[1][2]],
            [bbox_3d[1][0], bbox_3d[0][1], bbox_3d[1][2]],
            [bbox_3d[0][0], bbox_3d[1][1], bbox_3d[1][2]],
            [bbox_3d[1][0], bbox_3d[1][1], bbox_3d[1][2]],
        ])
        points_2d = cam_K @ points_3d.T
        points_2d = points_2d[:2] / points_2d[2]  # Normalize by depth
        x_min, y_min = points_2d.min(axis=1)
        x_max, y_max = points_2d.max(axis=1)
        return [x_min, y_min, x_max, y_max]

    def compute_iou(self, box1, box2):
        """Compute Intersection over Union (IoU) between two 2D bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def filter_detections(self, detected_objects, min_area=500, max_area=50000):
        """Filter detected objects based on size and other heuristics."""
        filtered_objects = []
        for obj in detected_objects:
            x_min, y_min, x_max, y_max = obj['box']
            area = (x_max - x_min) * (y_max - y_min)
            if min_area <= area <= max_area:
                filtered_objects.append(obj)
            else:
                self.get_logger().info(f"Filtered out detection with area {area}.")
        return filtered_objects

    def process_images(self):
        if self.color_image is None or self.depth_image is None or self.cam_K is None:
            return

        H, W = self.color_image.shape[:2]
        color = cv2.resize(self.color_image, (W, H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(self.depth_image, (W, H), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.1) | (depth >= np.inf)] = 0

        if not self.initial_detection_done:
            # Perform object detection and registration only once
            self.get_logger().info("Performing initial object detection and registration.")
            res = self.seg_model.predict(color)[0]
            if not res:
                self.get_logger().warn("No masks detected by SAM.")
                return

            detected_objects = []
            for r in res:
                for c in r:
                    mask = np.zeros((H, W), np.uint8)
                    contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                    _ = cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                    detected_objects.append({
                        'mask': mask,
                        'box': c.boxes.xyxy.tolist().pop(),
                        'contour': contour
                    })

            if not detected_objects:
                self.get_logger().warn("No objects found in the image.")
                return

            # Filter detections to remove false positives
            detected_objects = self.filter_detections(detected_objects)

            # Match detected objects to STL files
            for obj in detected_objects:
                obj_box = obj['box']
                best_match_idx = None
                best_iou = 0

                for idx, bbox_3d in enumerate(self.bboxes):
                    bbox_2d = self.project_3d_bbox_to_2d(bbox_3d, self.cam_K)
                    iou = self.compute_iou(obj_box, bbox_2d)
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = idx

                if best_match_idx is not None and best_iou > 0.1:  # Threshold for a valid match
                    if best_match_idx not in self.pose_estimations:
                        mesh_to_assign = self.meshes[best_match_idx]
                        bounds_to_assign = self.bounds[best_match_idx]
                        to_origin, _ = bounds_to_assign

                        try:
                            pose_est = FoundationPose(
                                model_pts=mesh_to_assign.vertices,
                                model_normals=mesh_to_assign.vertex_normals,
                                mesh=mesh_to_assign,
                                scorer=self.scorer,
                                refiner=self.refiner,
                                glctx=self.glctx,
                                debug=1,
                                debug_dir='./foundationpose_debug'
                            )
                            pose = pose_est.register(K=self.cam_K, rgb=color, depth=depth, ob_mask=obj['mask'], iteration=args.est_refine_iter)
                            pose_est.is_register = True  # Ensure the flag is updated after registration
                            self.pose_estimations[best_match_idx] = {
                                'pose_est': pose_est,
                                'mask': obj['mask'],
                                'to_origin': to_origin
                            }
                            self.get_logger().info(f"Initialized tracking for object {best_match_idx}.")
                        except Exception as e:
                            self.get_logger().error(f"Error initializing FoundationPose for object {best_match_idx}: {e}")

            self.initial_detection_done = True  # Mark initial detection as completed
            self.get_logger().info("Initial detection and registration completed.")

        # Process tracking for all initialized objects
        visualization_image = np.copy(color)
        for idx, data in self.pose_estimations.items():
            pose_est = data['pose_est']
            to_origin = data['to_origin']

            if pose_est.is_register:
                # Perform tracking
                self.get_logger().info(f"Tracking object {idx}.")
                try:
                    pose = pose_est.track_one(rgb=color, depth=depth, K=self.cam_K, iteration=args.track_refine_iter)
                    center_pose = pose @ np.linalg.inv(to_origin)

                    self.publish_pose_stamped(center_pose, f"object_{idx}_frame", f"/Current_OBJ_position_{idx+1}")
                    visualization_image = self.visualize_pose(visualization_image, center_pose, idx)
                except RuntimeError:
                    self.get_logger().error(f"Tracking failed for object {idx}. Ensure it was registered properly.")
            else:
                self.get_logger().warn(f"Object {idx} is not registered. Skipping tracking.")

        cv2.imshow('Pose Estimation & Tracking', visualization_image[..., ::-1])
        cv2.waitKey(1)

    def visualize_pose(self, image, center_pose, idx):
        bbox = self.bboxes[idx % len(self.bboxes)]
        vis = draw_posed_3d_box(self.cam_K, img=image, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=self.cam_K, thickness=3, transparency=0, is_input_rgb=True)
        return vis

    def publish_pose_stamped(self, center_pose, frame_id, topic_name):
        if topic_name not in self.pose_publishers:
            self.pose_publishers[topic_name] = self.create_publisher(PoseStamped, topic_name, 10)
        # Convert the center_pose matrix to a PoseStamped message
        pose_stamped_msg = PoseStamped()
        pose_stamped_msg.header.stamp = self.get_clock().now().to_msg()
        pose_stamped_msg.header.frame_id = frame_id

        # Convert center_pose to the pose format
        position = center_pose[:3, 3]
        rotation_matrix = center_pose[:3, :3]
        quaternion = R.from_matrix(rotation_matrix).as_quat()

        # Combine position and quaternion into a single array
        pose_array = np.concatenate((position, quaternion))

        # Apply transformation to convert from camera to base frame
        transformed_pose = transformation(pose_array)

        # Populate PoseStamped message with transformed pose
        pose_stamped_msg.pose.position.x = transformed_pose[0]
        pose_stamped_msg.pose.position.y = transformed_pose[1]
        pose_stamped_msg.pose.position.z = transformed_pose[2]

        pose_stamped_msg.pose.orientation.w = transformed_pose[3]
        pose_stamped_msg.pose.orientation.x = transformed_pose[4]
        pose_stamped_msg.pose.orientation.y = transformed_pose[5]
        pose_stamped_msg.pose.orientation.z = transformed_pose[6]

        # Publish the transformed pose
        self.pose_publishers[topic_name].publish(pose_stamped_msg)

def main(args=None):
    # Get the absolute path to the demo_data directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    source_directory = os.path.join(script_dir, "demo_data")

    # Search for .obj and .stl files recursively
    file_paths = glob.glob(os.path.join(source_directory, '**', '*.obj'), recursive=True) + \
                 glob.glob(os.path.join(source_directory, '**', '*.stl'), recursive=True) + \
                 glob.glob(os.path.join(source_directory, '**', '*.STL'), recursive=True)
    # Log the loaded file paths
    logging.info(f"Found 3D files: {file_paths}")

    # Check if files exist
    if not file_paths:
        logging.error("No 3D files found in the demo_data directory.")
        return

    # Verify each file exists
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
        else:
            logging.info(f"File found: {file_path}")

    # Call the function to rearrange files through the GUI
    new_file_paths = rearrange_files(file_paths)

    rclpy.init(args=args)
    node = PoseEstimationNode(new_file_paths)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
