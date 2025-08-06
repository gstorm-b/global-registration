import open3d as o3d
import numpy as np
import argparse
import os
from scipy.spatial import ConvexHull, KDTree

def mesh_sampling(mesh_path: str,
                  num_samples: int = 100000,
                  leaf_size: float = 0.001,
                  write_pcd: bool = True) -> o3d.geometry.PointCloud:
    """
    Load mesh, uniformly sample points, voxel downsample, optionally save PCD.
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh: {mesh_path}")
    mesh.compute_triangle_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=num_samples)
    pcd = pcd.voxel_down_sample(voxel_size=leaf_size)
    if write_pcd:
        out = os.path.splitext(mesh_path)[0] + ".pcd"
        o3d.io.write_point_cloud(out, pcd)
        print(f"Sampled point cloud saved to: {out}")
    return pcd


def HPR(pcd: o3d.geometry.PointCloud,
        camera_pos: np.ndarray,
        param: int) -> o3d.geometry.PointCloud:
    """
    Hidden Point Removal following PCL implementation:
      - For each point, compute P = p + 2*(R - ||p||)*p/||p||
      - Compute convex hull of P
      - For each hull vertex, find nearest original point
      - Return those visible points
    Args:
      pcd: input point cloud
      camera_pos: (3,) array for camera position
      param: exponent factor, R = max_norm * 10^param
    """
    pts = np.asarray(pcd.points)
    # translate so camera at origin
    p = pts - camera_pos
    norms = np.linalg.norm(p, axis=1)
    max_norm = norms.max()
    R = max_norm * (10 ** param)
    # compute P
    P = p + 2 * ((R - norms)[:, None] * p / norms[:, None])
    # build convex hull
    hull = ConvexHull(P)
    hull_vertices = P[hull.vertices]
    # KDTree on P
    tree = KDTree(P)
    # find nearest neighbors
    idx = set()
    for hv in hull_vertices:
        i = tree.query(hv, k=1)[1]
        idx.add(i)
    idx = list(idx)
    visible = pts[idx]
    visible_pcd = o3d.geometry.PointCloud()
    visible_pcd.points = o3d.utility.Vector3dVector(visible)
    return visible_pcd


def center_and_rotate(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Center at origin and rotate by quaternion [w,x,y,z]=[0.7071,0,-0.7071,0]"""
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    pts -= centroid
    R = o3d.geometry.get_rotation_matrix_from_quaternion([0.7071, 0, -0.7071, 0])
    pts = pts.dot(R.T)
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def main():
    parser = argparse.ArgumentParser(
        description="MeshSampling + Multi-view HPR + Centering"
    )
    parser.add_argument("-i","--input", required=True,
                        help="Input mesh file (.stl/.ply/.obj)")
    parser.add_argument("-n","--num_samples", type=int, default=1000000)
    parser.add_argument("-l","--leaf_size", type=float, default=0.005)
    parser.add_argument("-p","--param", type=int, default=3,
                        help="HPR parameter for R scaling (10^param)")
    parser.add_argument("--no_write", action='store_true')
    args = parser.parse_args()

    # step1: uniform sampling
    model_sampling = mesh_sampling(args.input, args.num_samples,
                                   args.leaf_size, write_pcd=False)
    print(f"Step1: Sampled {len(model_sampling.points)} points")

    # compute bounding cube center
    pts = np.asarray(model_sampling.points)
    min_pt = pts.min(axis=0)
    max_pt = pts.max(axis=0)
    center = (min_pt + max_pt) / 2
    half = np.max(max_pt - min_pt)
    # camera positions on cube faces (use numpy arrays for vector arithmetic)
    directions = np.array([[0,-1,0],[1,0,0],[0,1,0],[-1,0,0],[0,0,1],[0,0,-1]])
    cameras = [center + dir_vec * half for dir_vec in directions]

    # step2: HPR from multiple views
    combined = o3d.geometry.PointCloud()
    for i, cam in enumerate(cameras):
        print(f"HPR view {i} at camera {cam}")
        vis = HPR(model_sampling, np.array(cam), args.param)
        combined += vis
    print(f"Combined visible points: {len(combined.points)}")

    # step3: center and rotate
    final = center_and_rotate(combined)
    print("Centered and rotated model")

    if not args.no_write:
        out = os.path.splitext(args.input)[0] + "_hpr.pcd"
        o3d.io.write_point_cloud(out, final)
        print(f"Saved final PCD: {out}")

    o3d.visualization.draw_geometries([final], window_name='HPR Result')

if __name__ == '__main__':
    main()

# python mesh_sampling.py -i data/K41144.stl -n 1000000 -l 0.001