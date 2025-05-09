import os
import glob
import numpy as np
import trimesh
import open3d as o3d
import cv2


def trimesh_to_open3d(mesh):
    """
    Convert a trimesh mesh to an Open3D mesh.

    Args:
        mesh: A trimesh mesh object

    Returns:
        mesh_o3d: An Open3D TriangleMesh object
    """
    # Extract mesh components from trimesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    
    # Create new Open3D mesh
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d


def render_silhouettes_for_mesh(
    mesh_path,
    steps=10,
    num_views=30,
    x_range=(0, 90),
    y_range=(0, 90),
    z_range=(0, 90)
):
    """
    Render silhouettes for a 3D mesh from multiple viewpoints.

    Args:
        mesh_path: Path to the mesh file
        steps: Number of steps for angle sampling
        num_views: Maximum number of views to render
        x_range: Range of angles for x-axis rotation in degrees (min, max)
        y_range: Range of angles for y-axis rotation in degrees (min, max)
        z_range: Range of angles for z-axis rotation in degrees (min, max)
    """
    # Load mesh and prepare it for rendering
    mesh = trimesh.load(mesh_path)
    mesh_o3d = trimesh_to_open3d(mesh)
    mesh_o3d.paint_uniform_color([1, 1, 1])  # White color for better silhouette extraction
    mesh_o3d.compute_vertex_normals()
    img_width, img_height = 640, 480  # Standard resolution for rendered images

    # Target directory for silhouettes - creates a subdirectory with mesh name
    mesh_dir = os.path.dirname(mesh_path)
    mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
    sil_dir = os.path.join(mesh_dir, f"{mesh_name}_silhouettes")
    os.makedirs(sil_dir, exist_ok=True)

    # Generate a grid of angle combinations for multi-view rendering
    angles_x = np.linspace(x_range[0], x_range[1], steps, endpoint=False)
    angles_y = np.linspace(y_range[0], y_range[1], steps, endpoint=False)
    angles_z = np.linspace(z_range[0], z_range[1], steps, endpoint=False)
    all_angles = [(x, y, z) for x in angles_x for y in angles_y for z in angles_z]
    
    # Select a subset of angles if there are too many combinations
    np.random.seed(42)  # Fixed seed for reproducibility
    if len(all_angles) > num_views:
        selected_angles = np.random.choice(len(all_angles), num_views, replace=False)
        angle_combos = [all_angles[i] for i in selected_angles]
    else:
        angle_combos = all_angles

    # Set up the Open3D visualizer for off-screen rendering
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=img_width, height=img_height)
    vis.add_geometry(mesh_o3d)
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)  # Adjust zoom to ensure object is fully visible
    
    # Configure rendering options for clean silhouettes
    opt = vis.get_render_option()
    opt.light_on = False  # Turn off lighting for cleaner silhouettes
    opt.background_color = np.array([0, 0, 0])  # Black background
    render_count = 0

    # Render silhouettes from different angles
    for x, y, z in angle_combos:
        # Create a copy of the mesh for this specific rotation
        mesh_render = o3d.geometry.TriangleMesh(mesh_o3d)
        mesh_render.paint_uniform_color([1, 1, 1])
        mesh_render.compute_vertex_normals()
        
        # Calculate rotation matrices for each axis
        R_x = mesh_render.get_rotation_matrix_from_axis_angle(np.deg2rad(x) * np.array([1, 0, 0]))
        R_y = mesh_render.get_rotation_matrix_from_axis_angle(np.deg2rad(y) * np.array([0, 1, 0]))
        R_z = mesh_render.get_rotation_matrix_from_axis_angle(np.deg2rad(z) * np.array([0, 0, 1]))
        
        # Apply rotations around mesh center
        mesh_render.rotate(R_x, center=mesh_render.get_center())
        mesh_render.rotate(R_y, center=mesh_render.get_center())
        mesh_render.rotate(R_z, center=mesh_render.get_center())
        
        # Update visualizer with rotated mesh
        vis.clear_geometries()
        vis.add_geometry(mesh_render)
        vis.poll_events()
        vis.update_renderer()
        
        # Capture rendered image and save it
        sil_path = os.path.join(sil_dir, f"x{x:.0f}_y{y:.0f}_z{z:.0f}.png")
        vis.capture_screen_image(sil_path)
        
        # Post-process the image to extract clean silhouette
        img = cv2.imread(sil_path)
        if img is not None:
            # Create binary mask where white pixels (object) > threshold
            mask = np.all(img > [10, 10, 10], axis=2).astype(np.uint8) * 255
            mask = cv2.merge([mask, mask, mask])  # Convert to 3-channel for saving
            cv2.imwrite(sil_path, mask)
        render_count += 1
    
    vis.destroy_window()
    print(f"Rendered {render_count} silhouettes for {mesh_path} in {sil_dir}")


def main():
    """
    Main function to process all mesh files in the demo_data directory.
    """
    # Find the base directory and locate all mesh files
    base_dir = os.path.dirname(os.path.realpath(__file__))
    demo_data_dir = os.path.join(base_dir, "demo_data")
    
    # Collect all supported mesh formats (obj, stl)
    mesh_files = glob.glob(os.path.join(demo_data_dir, '**', '*.obj'), recursive=True) + \
                 glob.glob(os.path.join(demo_data_dir, '**', '*.stl'), recursive=True) + \
                 glob.glob(os.path.join(demo_data_dir, '**', '*.STL'), recursive=True)
    
    # Process each mesh file
    for mesh_path in mesh_files:
        render_silhouettes_for_mesh(mesh_path)


if __name__ == "__main__":
    main()
