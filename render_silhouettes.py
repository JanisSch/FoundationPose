import os
import glob
import numpy as np
import trimesh
import open3d as o3d
import cv2

def trimesh_to_open3d(mesh):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d

def render_silhouettes_for_mesh(mesh_path, steps=10, num_views=30, x_range=(0, 90), y_range=(0, 90), z_range=(0, 90)):
    mesh = trimesh.load(mesh_path)
    mesh_o3d = trimesh_to_open3d(mesh)
    mesh_o3d.paint_uniform_color([1, 1, 1])
    mesh_o3d.compute_vertex_normals()
    img_width, img_height = 640, 480

    # Zielordner für Silhouetten
    mesh_dir = os.path.dirname(mesh_path)
    mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
    sil_dir = os.path.join(mesh_dir, f"{mesh_name}_silhouettes")
    os.makedirs(sil_dir, exist_ok=True)

    # Winkelkombinationen
    angles_x = np.linspace(x_range[0], x_range[1], steps, endpoint=False)
    angles_y = np.linspace(y_range[0], y_range[1], steps, endpoint=False)
    angles_z = np.linspace(z_range[0], z_range[1], steps, endpoint=False)
    all_angles = [(x, y, z) for x in angles_x for y in angles_y for z in angles_z]
    np.random.seed(42)
    if len(all_angles) > num_views:
        selected_angles = np.random.choice(len(all_angles), num_views, replace=False)
        angle_combos = [all_angles[i] for i in selected_angles]
    else:
        angle_combos = all_angles

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=img_width, height=img_height)
    vis.add_geometry(mesh_o3d)
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    opt = vis.get_render_option()
    opt.light_on = False
    opt.background_color = np.array([0, 0, 0])
    render_count = 0

    for x, y, z in angle_combos:
        mesh_render = o3d.geometry.TriangleMesh(mesh_o3d)
        mesh_render.paint_uniform_color([1, 1, 1])
        mesh_render.compute_vertex_normals()
        R_x = mesh_render.get_rotation_matrix_from_axis_angle(np.deg2rad(x) * np.array([1, 0, 0]))
        R_y = mesh_render.get_rotation_matrix_from_axis_angle(np.deg2rad(y) * np.array([0, 1, 0]))
        R_z = mesh_render.get_rotation_matrix_from_axis_angle(np.deg2rad(z) * np.array([0, 0, 1]))
        mesh_render.rotate(R_x, center=mesh_render.get_center())
        mesh_render.rotate(R_y, center=mesh_render.get_center())
        mesh_render.rotate(R_z, center=mesh_render.get_center())
        vis.clear_geometries()
        vis.add_geometry(mesh_render)
        vis.poll_events()
        vis.update_renderer()
        sil_path = os.path.join(sil_dir, f"x{x:.0f}_y{y:.0f}_z{z:.0f}.png")
        vis.capture_screen_image(sil_path)
        img = cv2.imread(sil_path)
        if img is not None:
            mask = np.all(img > [10, 10, 10], axis=2).astype(np.uint8) * 255
            mask = cv2.merge([mask, mask, mask])
            cv2.imwrite(sil_path, mask)
        render_count += 1
    vis.destroy_window()
    print(f"Rendered {render_count} silhouettes for {mesh_path} in {sil_dir}")

def main():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    demo_data_dir = os.path.join(base_dir, "demo_data")
    mesh_files = glob.glob(os.path.join(demo_data_dir, '**', '*.obj'), recursive=True) + \
                 glob.glob(os.path.join(demo_data_dir, '**', '*.stl'), recursive=True) + \
                 glob.glob(os.path.join(demo_data_dir, '**', '*.STL'), recursive=True)
    for mesh_path in mesh_files:
        render_silhouettes_for_mesh(mesh_path)

if __name__ == "__main__":
    main()
