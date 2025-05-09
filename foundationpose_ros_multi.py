import sys
import os
import glob
import logging
import argparse
import itertools
import yaml

# Set sys.path relative to the script location
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(code_dir, 'FoundationPose'))
sys.path.append(os.path.join(code_dir, 'FoundationPose', 'nvdiffrast'))
sys.path.append(code_dir)
sys.path.append(os.path.join(code_dir, 'FoundationPose'))

# Third-party imports
import numpy as np
import cv2
import trimesh
import scipy.optimize
import scipy.ndimage
import open3d as o3d
import tkinter as tk
from tkinter import Button
from scipy.spatial.transform import Rotation as R
from ultralytics import SAM

# ROS2 imports - Explicitly import message types to avoid ROS1/ROS2 conflicts
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROS2Image
from sensor_msgs.msg import CameraInfo as ROS2CameraInfo
from geometry_msgs.msg import Pose as ROS2Pose
from geometry_msgs.msg import PoseStamped as ROS2PoseStamped
from geometry_msgs.msg import TransformStamped as ROS2TransformStamped
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Bool as ROS2Bool

# Local imports
from estimater import *
from cam_2_base_transform import *

# Configure logging
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------------------------
# Monkey patching FoundationPose to add tracking capabilities
# --------------------------------------------------------------------------

# Save the original `__init__` and `register` methods
original_init = FoundationPose.__init__
original_register = FoundationPose.register


def modified_init(self, model_pts, model_normals, symmetry_tfs=None, mesh=None,
                  scorer=None, refiner=None, glctx=None, debug=0,
                  debug_dir='./FoundationPose'):
    """
    Modified __init__ to add is_register attribute.
    
    This monkey patch adds an attribute to track registration status.
    
    Args:
        model_pts: Model points
        model_normals: Model normals
        symmetry_tfs: Symmetry transformations (optional)
        mesh: Mesh object (optional)
        scorer: Score predictor (optional)
        refiner: Pose refiner (optional)
        glctx: Graphics context (optional)
        debug: Debug level (optional)
        debug_dir: Debug directory (optional)
    """
    original_init(self, model_pts, model_normals, symmetry_tfs, mesh, scorer,
                  refiner, glctx, debug, debug_dir)
    self.is_register = False  # Initialize as False


def modified_register(self, K, rgb, depth, ob_mask, iteration):
    """
    Modified register to set is_register to True when a pose is registered.
    
    This monkey patch updates the registration status when a pose is successfully registered.
    
    Args:
        K: Camera intrinsic matrix
        rgb: RGB image
        depth: Depth image
        ob_mask: Object mask
        iteration: Number of refinement iterations
        
    Returns:
        numpy.ndarray: 4x4 pose matrix
    """
    pose = original_register(self, K, rgb, depth, ob_mask, iteration)
    self.is_register = True  # Set to True after registration
    return pose


# Apply the monkey patches
FoundationPose.__init__ = modified_init
FoundationPose.register = modified_register


# --------------------------------------------------------------------------
# GUI for object selection
# --------------------------------------------------------------------------

class FileSelectorGUI:
    """
    GUI for selecting files and specifying max occurrences for each object.
    
    This class creates a tkinter GUI that allows users to select which objects
    to track and specify the maximum number of instances of each object that
    might appear in the scene.
    """
    def __init__(self, master, file_paths):
        """
        Initialize the file selector GUI.
        
        Args:
            master: Tkinter root window
            file_paths: List of file paths to select from
        """
        self.master = master
        self.master.title("Library: Sequence Selector")
        self.file_paths = file_paths
        self.selected_objects = []  # Store selected objects and their max counts
        self.check_vars = []  # Store variables for checkboxes and max counts

        # Add a label to guide the user
        self.label = tk.Label(
            master,
            text="Select objects to track and specify their maximum occurrences:"
        )
        self.label.pack(pady=5)

        # Create a frame to hold the checkboxes and input fields
        self.checkbox_frame = tk.Frame(master)
        self.checkbox_frame.pack()

        # Create a checkbox and input field for each file
        for file_path in self.file_paths:
            var = tk.BooleanVar()
            max_count_var = tk.StringVar(value="1")  # Default max count is 1
            self.check_vars.append((var, max_count_var, file_path))

            file_name = os.path.splitext(os.path.basename(file_path))[0]
            row_frame = tk.Frame(self.checkbox_frame)
            row_frame.pack(anchor="w", pady=2)

            checkbox = tk.Checkbutton(
                row_frame,
                text=file_name,
                variable=var,
                command=lambda v=var, rf=row_frame: self.toggle_max_field(v, rf)
            )
            checkbox.pack(side="left")

            max_count_label = tk.Label(row_frame, text="Max:")
            max_count_label.pack(side="left", padx=5)
            max_count_label.pack_forget()  # Initially hide the label

            max_count_entry = tk.Entry(
                row_frame, textvariable=max_count_var, width=5
            )
            max_count_entry.pack(side="left")
            max_count_entry.pack_forget()  # Initially hide the entry

            # Store references to the label and entry for toggling
            row_frame.max_count_label = max_count_label
            row_frame.max_count_entry = max_count_entry

        # Add a "Done" button to confirm the selection
        self.done_button = Button(master, text="Done", command=self.done)
        self.done_button.pack(pady=10)

    def toggle_max_field(self, var, row_frame):
        """
        Show or hide the Max field based on the checkbox state.
        
        Args:
            var: Boolean variable for the checkbox
            row_frame: Frame containing the max count field
        """
        if var.get():
            row_frame.max_count_label.pack(side="left", padx=5)
            row_frame.max_count_entry.pack(side="left")
        else:
            row_frame.max_count_label.pack_forget()
            row_frame.max_count_entry.pack_forget()

    def done(self):
        """
        Save the selected objects and their max counts, then close the GUI.
        """
        self.selected_objects = [
            {"file_path": file_path, "max_count": int(max_count_var.get())}
            for var, max_count_var, file_path in self.check_vars if var.get()
        ]
        self.master.quit()

    def get_selected_objects(self):
        """
        Return the selected objects and their max counts after the GUI has closed.
        
        Returns:
            List of dictionaries with file_path and max_count
        """
        return self.selected_objects


def rearrange_files(file_paths):
    """
    Launch the file selector GUI and return the selected objects.
    
    Args:
        file_paths: List of file paths to select from
        
    Returns:
        List of selected objects with their max counts
    """
    root = tk.Tk()
    app = FileSelectorGUI(root, file_paths)
    root.mainloop()  # Start the GUI event loop
    selected_objects = app.get_selected_objects()  # Get selected objects
    
    # Safely destroy the Tkinter root window
    try:
        if root.winfo_exists():
            root.destroy()  # Only try to destroy if window still exists
    except tk.TclError:
        # Window already destroyed, ignore the error
        pass
        
    return selected_objects


# --------------------------------------------------------------------------
# Command-line arguments
# --------------------------------------------------------------------------

# Argument Parser
parser = argparse.ArgumentParser(description="FoundationPose ROS2 node for multi-object pose estimation and tracking")
code_dir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument('--est_refine_iter', type=int, default=4,
                   help='Number of refinement iterations for initial pose estimation')
parser.add_argument('--track_refine_iter', type=int, default=2,
                   help='Number of refinement iterations for pose tracking')
args = parser.parse_args()


# --------------------------------------------------------------------------
# Main ROS2 Node
# --------------------------------------------------------------------------

class PoseEstimationNode(Node):
    """
    ROS2 Node for pose estimation and tracking of multiple objects.
    
    This node subscribes to camera image topics, performs object detection
    and pose estimation, and publishes the estimated poses through TF
    and dedicated topic publishers.
    """
    def __init__(self, selected_objects):
        """
        Initialize the pose estimation node.
        
        Args:
            selected_objects: List of dictionaries with file_path and max_count
        """
        super().__init__('pose_estimation_node')

        # Extract file paths and max counts from selected objects
        self.mesh_files = [obj["file_path"] for obj in selected_objects]
        self.max_counts = {
            obj["file_path"]: obj["max_count"] for obj in selected_objects
        }

        # ROS subscriptions and publishers - Use the renamed message types
        self.image_sub = self.create_subscription(
            ROS2Image, '/camera/camera/color/image_raw', self.image_callback, 10
        )
        self.depth_sub = self.create_subscription(
            ROS2Image, '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback, 10
        )
        self.info_sub = self.create_subscription(
            ROS2CameraInfo, '/camera/camera/color/camera_info',
            self.camera_info_callback, 10
        )

        # TF Broadcaster for publishing object poses in the TF tree
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribe to the /redo topic to reset detection
        self.redo_sub = self.create_subscription(
            ROS2Bool, '/redo', self.redo_callback, 10
        )

        # Initialize bridge and image containers
        self.bridge = CvBridge()
        self.depth_image = None
        self.color_image = None
        self.cam_K = None  # Initialize cam_K as None until we receive camera info

        # Load selected meshes
        self.meshes = [trimesh.load(mesh) for mesh in self.mesh_files]

        # Log loaded meshes
        logging.info(f"Selected 3D files for tracking: {self.meshes}")
        logging.info(f"Max counts for objects: {self.max_counts}")
        if not self.meshes:
            logging.error(
                "No 3D files were loaded into self.meshes. "
                "Check file paths and processing logic."
            )

        # Calculate bounding boxes for visualization
        self.bounds = [
            trimesh.bounds.oriented_bounds(mesh) for mesh in self.meshes
        ]
        self.bboxes = [
            np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
            for _, extents in self.bounds
        ]

        # Initialize FoundationPose components
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()

        # Initialize SAM2 model for segmentation
        self.seg_model = SAM("sam2.1_b.pt")

        # State tracking variables
        self.pose_estimations = {}  # Track multiple pose estimations
        self.pose_publishers = {}  # Store publishers for each object
        self.initial_selection_done = False  # Track if initial selection done
        self.all_meshes = self.meshes.copy()
        self.all_bounds = self.bounds.copy()
        self.all_bboxes = self.bboxes.copy()
        self.initial_detection_done = False  # Track if detection is completed

        # Silhouette and pose guess storage for initialization
        self.rendered_silhouettes = {}  # {mesh_idx: [mask1, mask2, ...]}
        self.pose_guesses = {}          # {mesh_idx: [pose1, pose2, ...]}

        # Load camera intrinsics from static_camera_info.yaml for rendering
        self.static_cam_K = self.load_static_camera_intrinsics()

        # Load silhouettes for selected meshes
        self.load_silhouettes_for_selected_meshes()

    # -------------------------------------------------------------------------
    # Initialization helper methods
    # -------------------------------------------------------------------------

    def load_static_camera_intrinsics(self):
        """
        Load camera intrinsics from static_camera_info.yaml for rendering.
        
        Returns:
            numpy.ndarray: Camera intrinsic matrix or None if not found
        """
        # Use static_camera_info.yaml one directory above script location
        yaml_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 
            "static_camera_info.yaml"
        )
        if not os.path.exists(yaml_path):
            self.get_logger().warn(
                f"Camera intrinsics file not found: {yaml_path}"
            )
            return None
            
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
            
        # Try to find the 'camera_matrix' or 'K' field
        if "camera_matrix" in data and "data" in data["camera_matrix"]:
            K = np.array(data["camera_matrix"]["data"]).reshape((3, 3))
        elif "K" in data:
            K = np.array(data["K"]).reshape((3, 3))
        else:
            self.get_logger().warn(
                "No camera matrix found in static_camera_info.yaml"
            )
            return None
            
        return K

    def load_silhouettes_for_selected_meshes(self):
        """
        Load silhouette images for all selected meshes from their subfolders.
        
        For each mesh, looks for a corresponding silhouettes folder containing
        pre-rendered views from different angles.
        """
        for mesh_idx, mesh_path in enumerate(self.mesh_files):
            mesh_dir = os.path.dirname(mesh_path)
            mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
            sil_dir = os.path.join(mesh_dir, f"{mesh_name}_silhouettes")
            sil_files = sorted(glob.glob(os.path.join(sil_dir, "*.png")))
            silhouettes = []
            poses = []
            
            for sil_file in sil_files:
                mask = cv2.imread(sil_file, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    silhouettes.append(mask)
                    # Extract the pose from the filename, e.g., x45_y45_z0.png
                    fname = os.path.basename(sil_file)
                    try:
                        # Extract pose angles from filename format x{angle}_y{angle}_z{angle}
                        x = float(fname.split("_")[0][1:])
                        y = float(fname.split("_")[1][1:])
                        z = float(fname.split("_")[2][1:].split(".")[0])
                        poses.append((x, y, z))
                    except Exception:
                        # Use default pose if filename format is unexpected
                        poses.append((0, 0, 0))
                        
            self.rendered_silhouettes[mesh_idx] = silhouettes
            self.pose_guesses[mesh_idx] = poses
            self.get_logger().info(
                f"Loaded {len(silhouettes)} silhouettes for mesh "
                f"{mesh_path} from {sil_dir}"
            )

    # -------------------------------------------------------------------------
    # ROS Callbacks
    # -------------------------------------------------------------------------

    def camera_info_callback(self, msg):
        """
        Callback for camera info topic to initialize camera intrinsics.
        
        Args:
            msg: CameraInfo message
        """
        if self.cam_K is None:
            self.cam_K = np.array(msg.k).reshape((3, 3))
            self.get_logger().info(
                f"Camera intrinsic matrix initialized: {self.cam_K}"
            )

    def image_callback(self, msg):
        """
        Callback for color image topic.
        
        Args:
            msg: Image message
        """
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.process_images()

    def depth_callback(self, msg):
        """
        Callback for depth image topic.
        
        Args:
            msg: Image message
        """
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1") / 1e3
        self.process_images()

    def redo_callback(self, msg):
        """
        Callback to handle /redo topic. Resets initial detection.
        
        Args:
            msg: Bool message
        """
        if msg.data:  # If the message data is True
            self.get_logger().info(
                "Redo signal received. Resetting initial detection."
            )
            self.initial_detection_done = False  # Reset the detection flag
            self.pose_estimations.clear()  # Clear existing pose estimations
            self.process_images()  # Restart the detection process

    # -------------------------------------------------------------------------
    # Utility and processing methods
    # -------------------------------------------------------------------------

    def trimesh_to_open3d(self, mesh_trimesh):
        """
        Convert trimesh mesh to Open3D mesh.
        
        Args:
            mesh_trimesh: Trimesh mesh
            
        Returns:
            open3d.geometry.TriangleMesh: Open3D mesh
        """
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_trimesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_trimesh.faces)
        return mesh_o3d

    def transform_mask(self, mask, tx, ty, scale, angle, shape):
        """
        Transform a mask with translation, scaling, and rotation.
        
        Args:
            mask: Input mask
            tx: Translation in x
            ty: Translation in y
            scale: Scale factor
            angle: Rotation angle in degrees
            shape: Output shape
            
        Returns:
            numpy.ndarray: Transformed mask
        """
        # Create transformation matrix
        center = np.array(mask.shape) / 2
        M = cv2.getRotationMatrix2D((center[1], center[0]), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        
        # Apply transformation
        transformed = cv2.warpAffine(
            mask, M, (shape[0], shape[1]), flags=cv2.INTER_NEAREST
        )
        return transformed

    def render_synthetic_mask(self, mesh, cam_K, img_shape, pose_guess):
        """
        Render a binary mask of the mesh from the given pose using Open3D.
        Accepts both trimesh and open3d mesh objects.
        
        Args:
            mesh: Mesh object (trimesh or Open3D)
            cam_K: Camera intrinsic matrix
            img_shape: Output image shape
            pose_guess: Pose guess
            
        Returns:
            numpy.ndarray: Binary mask
        """
        # Convert mesh to Open3D format if needed
        if isinstance(mesh, o3d.geometry.TriangleMesh):
            mesh_o3d = mesh
        else:
            mesh_o3d = self.trimesh_to_open3d(mesh)
            
        width, height = img_shape[1], img_shape[0]
        fx, fy = cam_K[0, 0], cam_K[1, 1]
        cx, cy = cam_K[0, 2], cam_K[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )
        
        # Set up extrinsic (pose_guess)
        extrinsic = np.eye(4)
        if pose_guess is not None:
            extrinsic[:3, :] = pose_guess[:3, :]
            
        # Renderer
        render = o3d.visualization.rendering.OffscreenRenderer(width, height)
        render.scene.set_background([0, 0, 0, 0])
        render.scene.add_geometry(
            "obj", mesh_o3d, o3d.visualization.rendering.MaterialRecord()
        )
        render.setup_camera(intrinsic, extrinsic)
        mask = np.asarray(render.render_to_image())[..., 0] > 0  # Binary mask
        del render  # Ensure renderer is cleaned up
        
        return mask.astype(np.uint8) * 255

    def mask_similarity_sift(self, mask1, mask2):
        """
        Compute similarity between two binary masks using SIFT keypoints
        and descriptors. Downscale masks for faster SIFT computation.
        
        Args:
            mask1: First mask
            mask2: Second mask
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Downscale masks (e.g., 50% of original size)
        scale_factor = 0.5
        mask1_small = cv2.resize(
            mask1, (0, 0), fx=scale_factor, fy=scale_factor, 
            interpolation=cv2.INTER_NEAREST
        )
        mask2_small = cv2.resize(
            mask2, (0, 0), fx=scale_factor, fy=scale_factor, 
            interpolation=cv2.INTER_NEAREST
        )

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors for both masks
        keypoints1, descriptors1 = sift.detectAndCompute(mask1_small, None)
        keypoints2, descriptors2 = sift.detectAndCompute(mask2_small, None)

        if descriptors1 is None or descriptors2 is None:
            return 0

        # Use BFMatcher to find the best matches between descriptors
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        similarity = len(matches) / max(len(keypoints1), len(keypoints2))
        return similarity

    def filter_detections(self, detected_objects, min_area=0, max_area=200000,
                          min_iou_with_silhouette=0.05):
        """
        Filter detected objects based on mask area, optionally IoU with any
        silhouette, and other heuristics.
        Keeps objects whose mask area is within [min_area, max_area] and which
        have at least some overlap with any rendered silhouette.
        
        Args:
            detected_objects: List of detected objects
            min_area: Minimum mask area
            max_area: Maximum mask area
            min_iou_with_silhouette: Minimum IoU with any silhouette
            
        Returns:
            List of filtered objects
        """
        filtered_objects = []
        for obj in detected_objects:
            mask = obj['mask']
            area = np.sum(mask > 0)
            # Area filter
            if not (min_area <= area <= max_area):
                self.get_logger().info(
                    f"Filtered out detection with area {area}."
                )
                continue
                
            # IoU with any silhouette filter (to remove wall/false positives)
            has_silhouette_match = False
            for mesh_idx, silhouettes in self.rendered_silhouettes.items():
                for sil in silhouettes:
                    similarity = self.mask_similarity_sift(mask, sil)
                    if similarity > min_iou_with_silhouette:
                        has_silhouette_match = True
                        break
                if has_silhouette_match:
                    break
                    
            if not has_silhouette_match:
                self.get_logger().info(
                    f"Filtered out detection with area {area} due to no "
                    f"silhouette similarity > {min_iou_with_silhouette}."
                )
                continue
                
            filtered_objects.append(obj)
            
        self.get_logger().info(
            f"Detections after filtering: {len(filtered_objects)}"
        )
        return filtered_objects

    def best_iou_with_transform(self, rendered_mask, camera_mask):
        """
        Find the best IoU between rendered_mask and camera_mask by optimizing
        translation and rotation (no scaling).
        
        Args:
            rendered_mask: Rendered mask to transform
            camera_mask: Target camera mask
            
        Returns:
            tuple: (best_iou, best_params)
        """
        def neg_iou(params):
            tx, ty, angle = params
            transformed = self.transform_mask(
                rendered_mask, tx, ty, 1.0, angle, camera_mask.shape[::-1]
            )
            intersection = np.logical_and(
                transformed > 0, camera_mask > 0
            ).sum()
            union = np.logical_or(transformed > 0, camera_mask > 0).sum()
            if union == 0:
                return 1.0  # worst case
            return 1.0 - (intersection / union)

        # Initial guess: no translation, angle=0
        x0 = [0, 0, 0]
        bounds = [(-50, 50), (-50, 50), (-45, 45)]
        result = scipy.optimize.minimize(
            neg_iou, x0, bounds=bounds, method='L-BFGS-B'
        )
        best_params = result.x
        best_iou = 1.0 - result.fun
        return best_iou, best_params

    # -------------------------------------------------------------------------
    # Main processing pipeline
    # -------------------------------------------------------------------------

    def process_images(self):
        """
        Main processing function for handling images, detection, registration,
        and tracking.
        
        This is the core function that handles the entire pipeline:
        1. Initial detection of objects using SAM segmentation
        2. Matching detected objects to 3D models
        3. Pose estimation for matched objects
        4. Continuous tracking of objects in subsequent frames
        """
        # Skip if we don't have all necessary data yet
        if (self.color_image is None or self.depth_image is None or 
                self.cam_K is None):
            return
            
        color = self.color_image
        depth = self.depth_image
        H, W = color.shape[:2]
        
        # Ensure depth is 3D and clean up invalid values
        if len(depth.shape) == 2:
            depth = depth[..., np.newaxis]
        depth[(depth < 0.1) | (depth >= 5.0) | np.isnan(depth)] = 5.0

        # ----------- Initial Detection Phase -----------
        if not self.initial_detection_done:
            # Check if silhouettes are ready
            if not self.rendered_silhouettes or any(
                    len(sils) == 0 for sils in self.rendered_silhouettes.values()
            ):
                self.get_logger().warn(
                    "Silhouettes not yet rendered. Waiting for camera info and rendering."
                )
                return
                
            self.get_logger().info(
                "Performing object detection and registration."
            )

            # 1. Perform SAM segmentation
            detected_objects = []
            res = self.seg_model.predict(color)[0]
            debug_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "debug_masks"
            )
            os.makedirs(debug_dir, exist_ok=True)
            mask_debug_paths = []
            all_sam_masks = []  # Collect all masks, even broken ones
            
            # Process SAM segmentation results
            if res:
                mask_counter = 0
                for r in res:
                    for c in r:
                        mask = np.zeros((H, W), np.uint8)
                        if c.masks and c.masks.xy:
                            contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                            _ = cv2.drawContours(
                                mask, [contour], -1, (255, 255, 255), cv2.FILLED
                            )
                            kernel = np.ones((3, 3), np.uint8)
                            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                            area = np.sum(mask > 0)
                            area_ratio = area / (H * W)
                            
                            # Filter: skip mask if more than 25% white (likely wall/table)
                            if area_ratio > 0.2:
                                self.get_logger().warn(
                                    f"Skipped SAM mask {mask_counter} "
                                    f"(area_ratio={area_ratio:.2f}) - likely wall/table."
                                )
                                mask_counter += 1
                                continue
                                
                            # Filter masks if too many pixels > 2m
                            if self.depth_image is not None:
                                mask_depths = self.depth_image[mask > 0]
                                valid_depths = mask_depths[(~np.isnan(mask_depths))]
                                if valid_depths.size > 0:
                                    ratio_far = np.sum(valid_depths > 1.5) / valid_depths.size
                                    if ratio_far > 0.3:  # e.g., 30% of pixels further than 1.5m
                                        self.get_logger().warn(
                                            f"Skipped SAM mask {mask_counter} "
                                            f"({ratio_far*100:.1f}% of valid depths > 1.5m, "
                                            f"n_valid={valid_depths.size}) - too far from camera."
                                        )
                                        mask_counter += 1
                                        continue
                                        
                            # Save each mask as debug image (only if not skipped)
                            mask_path = os.path.join(
                                debug_dir, f"sam_mask_{mask_counter}.png"
                            )
                            cv2.imwrite(mask_path, mask)
                            mask_debug_paths.append(mask_path)
                            all_sam_masks.append({
                                'mask': mask,
                                'box': (c.boxes.xyxy.tolist().pop() 
                                       if c.boxes and c.boxes.xyxy is not None 
                                       else None),
                                'contour': contour
                            })
                            mask_counter += 1
                        else:
                            self.get_logger().warn(
                                "Detected object has no mask coordinates."
                            )
                self.get_logger().info(
                    f"Saved {mask_counter} SAM masks to {debug_dir}"
                )

            # 2. Remove duplicate masks (IoU > 0.4)
            unique_masks = []
            for obj_i in all_sam_masks:
                mask_i = obj_i['mask']
                is_duplicate = False
                for obj_j in unique_masks:
                    mask_j = obj_j['mask']
                    intersection = np.logical_and(mask_i > 0, mask_j > 0).sum()
                    union = np.logical_or(mask_i > 0, mask_j > 0).sum()
                    iou = intersection / union if union > 0 else 0
                    if iou > 0.4:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_masks.append(obj_i)
            all_sam_masks = unique_masks

            # 3. Use all SAM masks for further processing
            detected_objects = all_sam_masks

            # 4. Hungarian matching: maximize sum of scores
            # Each mask gets assigned to exactly ONE mesh-instance
            iou_threshold = 0.1  # LOWER threshold for more robust matching!
            num_masks = len(detected_objects)
            num_meshes = len(self.mesh_files)
            max_counts = [
                self.max_counts[self.mesh_files[i]] for i in range(num_meshes)
            ]

            # Build mesh_instance_list: each mesh_idx appears max_count times
            mesh_instance_list = []
            for mesh_idx, count in enumerate(max_counts):
                for inst in range(count):
                    mesh_instance_list.append(mesh_idx)
            num_instances = len(mesh_instance_list)
            score_matrix = np.full(
                (num_masks, num_instances), -1e6, dtype=np.float32
            )

            # Collect IoU debug info for saving
            iou_debug_lines = []

            # 5. Calculate similarity scores between masks and silhouettes
            for mask_idx, obj in enumerate(detected_objects):
                for inst_idx, mesh_idx in enumerate(mesh_instance_list):
                    silhouettes = self.rendered_silhouettes[mesh_idx]
                    # Direct sequential approach, no batch function
                    ious = [
                        self.mask_similarity_sift(obj['mask'], sil) 
                        for sil in silhouettes
                    ]
                    ious_over = [iou for iou in ious if iou > iou_threshold]
                    max_iou = max(ious) if ious else 0
                    if ious_over:
                        avg_score = np.mean(ious_over)
                        weighted_score = avg_score * len(ious_over)
                        score_matrix[mask_idx, inst_idx] = weighted_score
                    iou_debug_lines.append(
                        f"mask_idx={mask_idx}, mesh_idx={mesh_idx}, "
                        f"inst_idx={inst_idx}, max_iou={max_iou:.4f}, "
                        f"avg_iou_over_thresh="
                        f"{np.mean(ious_over) if ious_over else 0:.4f}, "
                        f"count_over_thresh={len(ious_over)}, "
                        f"weighted_score={weighted_score if ious_over else 0:.4f}, "
                        f"ious={[round(i, 4) for i in ious]}"
                    )

            # 6. Save IoU debug info to file
            debug_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "debug_masks"
            )
            os.makedirs(debug_dir, exist_ok=True)
            iou_debug_path = os.path.join(debug_dir, "iou_scores_matching.txt")
            with open(iou_debug_path, "w") as f:
                for line in iou_debug_lines:
                    f.write(line + "\n")
            self.get_logger().info(
                f"Saved IoU matching debug info to {iou_debug_path}"
            )

            # 7. Hungarian matching (maximize sum of scores)
            # This ensures: each mask is assigned to at most one mesh instance, 
            # and each mesh instance to at most one mask
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(-score_matrix)

            # 8. Initialize pose estimators for matched objects
            self.pose_estimations.clear()
            any_registered = False
            used_masks = set()
            used_instances = set()
            for mask_idx, inst_idx in zip(row_ind, col_ind):
                # Only assign if the score is positive (or above a threshold)
                score = score_matrix[mask_idx, inst_idx]
                if score < 0:
                    continue
                if mask_idx in used_masks or inst_idx in used_instances:
                    continue  # Prevent duplicate assignments
                    
                used_masks.add(mask_idx)
                used_instances.add(inst_idx)
                mesh_idx = mesh_instance_list[inst_idx]
                instance_number = sum(
                    1 for i in range(inst_idx + 1) 
                    if mesh_instance_list[i] == mesh_idx
                )
                mesh_to_assign = self.meshes[mesh_idx]
                bounds_to_assign = self.bounds[mesh_idx]
                to_origin, _ = bounds_to_assign
                silhouettes = self.rendered_silhouettes[mesh_idx]
                ious = [
                    self.mask_similarity_sift(detected_objects[mask_idx]['mask'], mask) 
                    for mask in silhouettes
                ]
                best_pose_idx = int(np.argmax(ious))
                pose_guess = self.pose_guesses[mesh_idx][best_pose_idx]
                try:
                    # Initialize FoundationPose for this object
                    pose_est = FoundationPose(
                        model_pts=mesh_to_assign.vertices,
                        model_normals=mesh_to_assign.vertex_normals,
                        mesh=mesh_to_assign,
                        scorer=self.scorer,
                        refiner=self.refiner,
                        glctx=self.glctx,
                        debug=1,
                    )
                    # Perform initial registration
                    pose = pose_est.register(
                        K=self.cam_K, 
                        rgb=color, 
                        depth=depth[..., 0], 
                        ob_mask=detected_objects[mask_idx]['mask'], 
                        iteration=args.est_refine_iter
                    )
                    pose_est.is_register = True
                    instance_key = f"{mesh_idx}_{instance_number}"
                    self.pose_estimations[instance_key] = {
                        'pose_est': pose_est,
                        'mask': detected_objects[mask_idx]['mask'],
                        'to_origin': to_origin,
                        'mesh_idx': mesh_idx
                    }
                    self.get_logger().info(
                        f"Assigned mask {mask_idx} to mesh "
                        f"{self.mesh_files[mesh_idx]} (instance {instance_number}) "
                        f"with avg IoU {score:.2f}"
                    )
                    any_registered = True
                except Exception as e:
                    self.get_logger().error(
                        f"Error initializing FoundationPose for "
                        f"object {mesh_idx}: {e}", 
                        exc_info=True
                    )

            # 9. Mark detection as complete
            self.initial_detection_done = True
            self.get_logger().info(
                "Initial detection and registration attempt completed."
            )
            if not any_registered:
                self.get_logger().warn(
                    "No objects were successfully registered during initial detection."
                )

            # 10. DEBUG: Save image with all detected masks and assigned objects
            debug_img = np.copy(color)
            color_list = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (255, 0, 255), (0, 255, 255),
                (128, 128, 0), (128, 0, 128), (0, 128, 128),
                (128, 128, 128), (255, 128, 0), (0, 128, 255)
            ]
            for idx, (instance_key, data) in enumerate(self.pose_estimations.items()):
                mask = data['mask']
                mesh_idx = data['mesh_idx']
                file_path = self.mesh_files[mesh_idx]
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                color_val = color_list[idx % len(color_list)]
                # Overlay mask
                mask_rgb = np.zeros_like(debug_img)
                for c in range(3):
                    mask_rgb[..., c] = (mask > 0) * color_val[c]
                debug_img = cv2.addWeighted(debug_img, 1.0, mask_rgb, 0.5, 0)
                # Draw instance label and object name
                M = cv2.moments(mask)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = f"{instance_key} ({file_name})"
                    cv2.putText(
                        debug_img, label, (cx, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_val, 2
                    )
            debug_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "debug_masks"
            )
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, "foundationpose_init_debug.png")
            cv2.imwrite(debug_path, debug_img[..., ::-1])  # Save as BGR for viewing
            self.get_logger().info(f"Saved initialization debug image to {debug_path}")

        # ----------- Tracking Phase -----------
        # Process tracking for all initialized objects
        visualization_image = np.copy(color)
        for instance_key, data in self.pose_estimations.items():
            pose_est = data['pose_est']
            to_origin = data['to_origin']
            mesh_idx = data['mesh_idx']  # Get the original mesh index
            file_path = self.mesh_files[mesh_idx]
            file_name = os.path.splitext(os.path.basename(file_path))[0]

            if pose_est.is_register:
                # Perform tracking
                try:
                    # Track the object in the current frame
                    pose = pose_est.track_one(
                        rgb=color, 
                        depth=depth[..., 0], 
                        K=self.cam_K, 
                        iteration=args.track_refine_iter
                    )
                    # Transform to centered pose
                    center_pose = pose @ np.linalg.inv(to_origin)
                    # Create unique frame ID for this instance
                    frame_id = f"{instance_key}"  # e.g., "0_1", "1_1", "1_2"
                    # Publish the pose through TF
                    self.publish_tf(center_pose, frame_id)
                    # Create unique topic name for this instance
                    topic_name = f"/{file_name}_{instance_key}_position"
                    # Publish the pose as a PoseStamped message
                    self.publish_pose_stamped(center_pose, frame_id, topic_name)
                    # Visualize the pose in the image
                    visualization_image = self.visualize_pose(
                        visualization_image, center_pose, mesh_idx
                    )
                except RuntimeError as e:
                    self.get_logger().error(
                        f"Tracking failed for object instance {instance_key}. "
                        f"Error: {e}"
                    )
                except Exception as e:
                    self.get_logger().error(
                        f"Unexpected error during tracking for {instance_key}: {e}", 
                        exc_info=True
                    )

        # Check if any objects are being tracked
        if not self.pose_estimations:
            self.get_logger().warn(
                "No objects are being tracked. No visualization will be shown."
            )

        # Display visualization with error handling
        try:
            cv2.imshow('Pose Estimation & Tracking', visualization_image[..., ::-1])
            key = cv2.waitKey(1)
            if key == 27:  # ESC to close window
                cv2.destroyAllWindows()
        except Exception as e:
            self.get_logger().error(f"cv2.imshow or waitKey failed: {e}")

    # -------------------------------------------------------------------------
    # Visualization and publishing methods
    # -------------------------------------------------------------------------

    def visualize_pose(self, image, center_pose, mesh_idx):
        """
        Visualize the pose by drawing the 3D bounding box and axes.
        
        Args:
            image: Input image
            center_pose: Object pose
            mesh_idx: Mesh index
            
        Returns:
            numpy.ndarray: Visualization image
        """
        bbox = self.bboxes[mesh_idx % len(self.bboxes)]
        vis = draw_posed_3d_box(
            self.cam_K, img=image, ob_in_cam=center_pose, bbox=bbox
        )
        vis = draw_xyz_axis(
            vis, ob_in_cam=center_pose, scale=0.1, K=self.cam_K, 
            thickness=3, transparency=0, is_input_rgb=True
        )
        return vis

    def publish_tf(self, center_pose, frame_id, parent_frame="world"):
        """
        Publish the object's pose as a TF frame.
        
        Args:
            center_pose: Object pose in camera frame
            frame_id: Name of the object frame
            parent_frame: Name of the parent frame
        """
        t = ROS2TransformStamped()  # Use the renamed message type
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = frame_id
        t.transform.translation.x = center_pose[0, 3]
        t.transform.translation.y = center_pose[1, 3]
        t.transform.translation.z = center_pose[2, 3]
        rotation_matrix = center_pose[:3, :3]
        quaternion = R.from_matrix(rotation_matrix).as_quat()
        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]

        self.tf_broadcaster.sendTransform(t)

    def publish_pose_stamped(self, center_pose, frame_id, topic_name):
        """
        Publish the pose as a PoseStamped message on a unique topic.
        
        Args:
            center_pose: Object pose
            frame_id: Frame ID
            topic_name: Topic name
        """
        # Create publisher if it doesn't exist yet
        if topic_name not in self.pose_publishers:
            self.pose_publishers[topic_name] = self.create_publisher(
                ROS2PoseStamped, topic_name, 10  # Use the renamed message type
            )
            self.get_logger().info(f"Created publisher for topic: {topic_name}")

        # Convert the center_pose matrix to a PoseStamped message
        pose_stamped_msg = ROS2PoseStamped()  # Use the renamed message type
        pose_stamped_msg.header.stamp = self.get_clock().now().to_msg()
        # Set the frame_id of the PoseStamped message to match the TF frame_id
        pose_stamped_msg.header.frame_id = "camera_color_optical_frame"  # Or your camera frame

        # Convert center_pose to the pose format
        position = center_pose[:3, 3]
        rotation_matrix = center_pose[:3, :3]
        quaternion = R.from_matrix(rotation_matrix).as_quat()
        # Combine position and quaternion into a single array
        pose_array = np.concatenate((position, quaternion))
        # Apply transformation to convert from camera to base frame
        transformed_pose = transformation(pose_array)  # Assuming this transforms to base_link or world
        # Populate PoseStamped message with transformed pose
        pose_stamped_msg.pose.position.x = position[0]
        pose_stamped_msg.pose.position.y = position[1]
        pose_stamped_msg.pose.position.z = position[2]
        pose_stamped_msg.pose.orientation.x = quaternion[0]
        pose_stamped_msg.pose.orientation.y = quaternion[1]
        pose_stamped_msg.pose.orientation.z = quaternion[2]
        pose_stamped_msg.pose.orientation.w = quaternion[3]

        # Publish the pose
        self.pose_publishers[topic_name].publish(pose_stamped_msg)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def main(args=None):
    """
    Main entry point for the ROS2 node.
    
    This function:
    1. Searches for 3D model files in the demo_data directory
    2. Launches a GUI for selecting which objects to track
    3. Initializes and runs the ROS2 node
    """
    # Get the absolute path to the demo_data directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    source_directory = os.path.join(script_dir, "demo_data")

    # Search for .obj and .stl files recursively
    file_paths = glob.glob(
        os.path.join(source_directory, '**', '*.obj'), recursive=True
    ) + glob.glob(
        os.path.join(source_directory, '**', '*.stl'), recursive=True
    ) + glob.glob(
        os.path.join(source_directory, '**', '*.STL'), recursive=True
    )

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

    # Call the function to rearrange files and select objects for tracking
    selected_objects = rearrange_files(file_paths)
    
    # Check if any objects were selected
    if not selected_objects:
        logging.warning("No objects were selected. Exiting.")
        return

    # Initialize and run the ROS2 node
    try:
        rclpy.init(args=args)
        node = PoseEstimationNode(selected_objects)
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except Exception as e:
        logging.error(f"Error in ROS2 node: {e}", exc_info=True)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()