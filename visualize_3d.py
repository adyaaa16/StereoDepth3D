import open3d as o3d
import time

ply_path = "data/output_3d/scene_model.ply"
print(f"Loading point cloud from: {ply_path} ...")
pcd = o3d.io.read_point_cloud(ply_path)
print(f"Loaded point cloud with {len(pcd.points)} vertices.")

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="StereoDepth3D - 3D Point Cloud View", width=1024, height=768, visible=True)

# Add geometry
vis.add_geometry(pcd)

# Configure rendering
opt = vis.get_render_option()
opt.point_size = 2.0
opt.background_color = [0.05, 0.05, 0.05]

# Reset view to ensure proper camera framing
vis.reset_view_point(True)

# Active event loop
print("3D View running! Click on the window to rotate/zoom. Press 'Q' or close window to exit.")
while True:
    vis.poll_events()
    vis.update_renderer()
    if not vis.poll_events():
        break
    time.sleep(0.01)

vis.destroy_window()