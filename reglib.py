
import open3d as o3d
import os

def mesh_to_point_cloud(stl_path: str,
                        number_of_points: int = 100000,
                        estimate_normals: bool = True,
                        normal_radius: float = 0.01) -> o3d.geometry.PointCloud:
    """
    Load a mesh from an STL file and sample a point cloud from its surface.

    Args:
        stl_path (str): Path to the input STL file.
        number_of_points (int): Number of points to sample on the mesh surface.
        estimate_normals (bool): Whether to estimate normals for the resulting point cloud.
        normal_radius (float): Radius to use for normal estimation (if enabled).

    Returns:
        o3d.geometry.PointCloud: The sampled point cloud.
    """
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(stl_path)
    if not mesh.has_triangles():
        raise ValueError(f"Failed to load mesh or mesh has no triangles: {stl_path}")

    # Sample points uniformly on the mesh
    pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)

    # Optionally estimate normals
    if estimate_normals:
        # Compute normals using k-nearest or radius search
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius,
                max_nn=30
            )
        )
        # Or orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=10)

    return pcd

def mesh_sampling(mesh_path: str,
                  num_samples: int = 100000,
                  leaf_size: float = 0.001,
                  write_pcd: bool = True) -> o3d.geometry.PointCloud:
    """
    Load a mesh (PLY/STL/OBJ/etc.), sample a point cloud uniformly from its surface,
    apply voxel downsampling, and optionally save as .pcd.

    Args:
        mesh_path: Path to the input mesh file.
        num_samples: Number of points to sample uniformly.
        leaf_size: Voxel size for downsampling.
        write_pcd: If True, writes the resulting cloud to <mesh_basename>.pcd.

    Returns:
        An Open3D PointCloud.
    """
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh or mesh is empty: {mesh_path}")

    # Ensure normals exist for sampling adaptation (optional)
    mesh.compute_triangle_normals()

    # Uniformly sample points
    pcd = mesh.sample_points_uniformly(number_of_points=num_samples)

    # Voxel downsample the point cloud
    pcd = pcd.voxel_down_sample(voxel_size=leaf_size)

    # Optionally write to .pcd file
    if write_pcd:
        output_path = os.path.splitext(mesh_path)[0] + ".pcd"
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Point cloud saved to: {output_path}")

    return pcd

def show_point_cloud(pcd_path: str):
    """
    Load and display a point cloud from a .pcd file using Open3D.

    Args:
        pcd_path (str): Path to the .pcd file.
    """
    # Read the point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        raise ValueError(f"Loaded point cloud is empty: {pcd_path}")

    # Print basic info
    print(f"Point Cloud loaded from: {pcd_path}")
    print(pcd)

    # Visualize
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=(0, 0, 0))

    o3d.visualization.draw_geometries([axis, pcd],
                                      window_name='Open3D - Point Cloud',
                                      width=800, height=600,
                                      left=50, top=50,
                                      point_show_normal=False)