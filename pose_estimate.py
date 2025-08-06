import open3d as o3d
import numpy as np
import argparse

def extract_visible_model(model: o3d.geometry.PointCloud,
                          scene: o3d.geometry.PointCloud,
                          num_candidates: int = 20,
                          radius_factor: float = 1.5) -> o3d.geometry.PointCloud:
    """
    Automatically infer the scene's camera viewpoint by sampling candidate directions
    and selecting the one yielding the largest visible subset of the model.
    Then extract visible points via Hidden Point Removal.
    """
    pts = np.asarray(model.points)
    scene_pts = np.asarray(scene.points)
    scene_center = scene_pts.mean(axis=0)
    radius = np.linalg.norm(scene_pts - scene_center, axis=1).max() * radius_factor

    best_idx = []
    max_visible = 0
    # Generate candidate directions on unit sphere
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    for i in range(num_candidates):
        theta = 2 * np.pi * i / phi
        z = 1 - (2 * i + 1) / num_candidates
        r = np.sqrt(1 - z*z)
        dir_vec = np.array([r * np.cos(theta), r * np.sin(theta), z])
        cam_pos = scene_center - dir_vec * radius
        # Hidden point removal
        _pts, idx = model.hidden_point_removal(cam_pos.tolist(), radius)
        if len(idx) > max_visible:
            max_visible = len(idx)
            best_idx = idx
    visible_model = model.select_by_index(best_idx)
    print(f"Chosen camera direction yields {max_visible} visible points.")
    return visible_model

def estimate_pose_global(model: o3d.geometry.PointCloud,
                         scene: o3d.geometry.PointCloud,
                         voxel_size: float = 0.02) -> o3d.pipelines.registration.RegistrationResult:
    """
    Estimate pose using Fast Global Registration on visible parts.
    """
    # Downsample
    m_down = model.voxel_down_sample(voxel_size)
    s_down = scene.voxel_down_sample(voxel_size)
    # Normals
    m_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 2, max_nn=30))
    s_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 2, max_nn=30))
    # FPFH features
    m_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        m_down, search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=100))
    s_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        s_down, search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=100))
    # Fast Global Registration
    option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=voxel_size * 1.5)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        m_down, s_down, m_fpfh, s_fpfh, option)
    return result

def refine_pose_icp(model: o3d.geometry.PointCloud,
                    scene: o3d.geometry.PointCloud,
                    initial_transform: np.ndarray,
                    voxel_size: float = 0.02) -> o3d.pipelines.registration.RegistrationResult:
    """
    Perform ICP refinement after global registration.
    """
    model.transform(initial_transform)

    # Ensure scene has normals
    if not scene.has_normals():
        scene.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=100))

    threshold = voxel_size * 1.0
    result = o3d.pipelines.registration.registration_icp(
        model, scene, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Global registration on visible model and scene"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--voxel", type=float, default=0.02)
    args = parser.parse_args()

    # Load PCDs
    model = o3d.io.read_point_cloud(args.model)
    scene = o3d.io.read_point_cloud(args.scene)
    if model.is_empty() or scene.is_empty():
        raise RuntimeError("Cannot load model or scene.")

    # Extract visible portion of model
    visible_model = extract_visible_model(model, scene)
    print(f"Visible model pts: {len(visible_model.points)}")

    # Visualize visible part vs scene
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.voxel * 5)
    print("Inspect visible model and scene. Close window to start registration.")
    o3d.visualization.draw_geometries([visible_model, scene, axis])

    # Estimate pose globally
    result = estimate_pose_global(visible_model, scene, voxel_size=args.voxel)
    print("Global Registration Result")
    print("Fitness:", result.fitness)
    print("Transformation:\n", result.transformation)

    # Apply transform for visual check
    visible_model.transform(result.transformation)
    print("Check alignment before ICP. Close window to continue.")
    o3d.visualization.draw_geometries([visible_model, scene, axis])

    # Refine using ICP
    print("Running ICP refinement...")
    model.transform(result.transformation)
    
    refined = refine_pose_icp(model, scene, np.identity(4), voxel_size=0.1)
    model.transform(refined.transformation)

    print("Final ICP Alignment Result")
    print("Fitness:", refined.fitness)
    print("Transformation:\n", refined.transformation)

    # Show final alignment
    o3d.visualization.draw_geometries([model, scene, axis])

if __name__ == '__main__':
    main()

# python pose_estimate.py --model data/K41144_hpr.pcd --scene data/view/view_01.pcd --voxel_size 0.002 --threshold 0.002
