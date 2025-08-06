import open3d as o3d
import numpy as np

def generate_synthetic_views(mesh_path, 
                             num_views=10, 
                             samples_per_view=50000,
                             noise_sigma=0.0001,
                             outlier_ratio=0.0001):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_triangle_normals()
    full_pcd = mesh.sample_points_uniformly(number_of_points=samples_per_view* num_views)
    
    # compute center and max radius
    pts = np.asarray(full_pcd.points)
    center = pts.mean(axis=0)
    radius = np.linalg.norm(pts - center, axis=1).max()
    
    synthetic_data = []
    for i in range(num_views):
        # random camera on a sphere surface
        theta = np.random.uniform(0, 2*np.pi)
        phi   = np.random.uniform(0, np.pi)
        cam = center + radius * np.array([
            np.sin(phi)*np.cos(theta),
            np.sin(phi)*np.sin(theta),
            np.cos(phi)
        ])
        # hidden point removal
        _, idx = full_pcd.hidden_point_removal(cam, radius*1.2)
        view_pcd = full_pcd.select_by_index(idx)
        
        # add gaussian noise
        pts_view = np.asarray(view_pcd.points)
        pts_view += np.random.normal(0, noise_sigma, size=pts_view.shape)
        
        # add random outliers
        n_out = int(len(pts_view) * outlier_ratio)
        outliers = np.random.uniform(
            low=pts_view.min(axis=0)-radius*0.1,
            high=pts_view.max(axis=0)+radius*0.1,
            size=(n_out, 3)
        )
        pts_view = np.vstack([pts_view, outliers])
        
        # create new PCD and record pose (here pose is identity, or you can randomize)
        synthetic = o3d.geometry.PointCloud()
        synthetic.points = o3d.utility.Vector3dVector(pts_view)
        
        # optional: random rotation + translation
        R = o3d.geometry.get_rotation_matrix_from_xyz(
            np.random.uniform(0, np.pi, size=3))
        t = np.random.uniform(-0.1, 0.1, size=3)
        synthetic.rotate(R, center=center)
        synthetic.translate(t)
        
        synthetic_data.append({
            "pcd": synthetic,
            "R": R,
            "t": t,
            "camera_pos": cam
        })
    return synthetic_data

# Usage example
data = generate_synthetic_views("data/K41144.stl")

for i, item in enumerate(data):
    o3d.io.write_point_cloud(f"data/view/view_{i:02d}.pcd", item["pcd"])