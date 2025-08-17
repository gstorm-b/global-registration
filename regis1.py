#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import copy
import argparse
import numpy as np
import open3d as o3d

# ========== Utils hình học & đối xứng ==========

def so3_geodesic_distance(Ra, Rb):
    """Khoảng cách geodesic trên SO(3)."""
    R = Ra @ Rb.T
    cos = (np.trace(R) - 1.0) * 0.5
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos)

def make_Cn_rotations(axis=np.array([0.0, 0.0, 1.0]), n=4):
    """Sinh nhóm đối xứng quay Cn quanh 'axis' (đã chuẩn hoá)."""
    axis = axis / np.linalg.norm(axis)
    G = []
    for k in range(n):
        theta = 2.0 * np.pi * k / n
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * theta)
        G.append(R)
    return G

def equivalent_under_symmetry(Ra, Rb, G, eps_deg=3.0):
    """Hai ma trận quay tương đương nếu chênh lệch nhỏ sau nhân 1 phần tử symmetry."""
    eps = np.deg2rad(eps_deg)
    for S in G:
        if so3_geodesic_distance(Ra, Rb @ S) < eps:
            return True
    return False

def deduplicate_hypotheses(hyps, G, eps_deg=3.0, tau=2e-3, topk=None):
    """
    hyps: list of dict { 'T':4x4, 'score':float }
    Khử trùng lặp theo symmetry + gần t. Giữ thứ tự theo score giảm dần.
    """
    kept = []
    for h in sorted(hyps, key=lambda x: -x['score']):
        T = h['T']
        R = T[:3, :3]
        t = T[:3, 3]
        ok = True
        for g in kept:
            Tg = g['T']
            Rg = Tg[:3, :3]
            tg = Tg[:3, 3]
            if np.linalg.norm(t - tg) < tau and equivalent_under_symmetry(R, Rg, G, eps_deg=eps_deg):
                ok = False
                break
        if ok:
            kept.append(h)
        if topk is not None and len(kept) >= topk:
            break
    return kept

# ========== I/O & tiền xử lý ==========

def load_point_cloud_any(path, voxel=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.pcd', '.ply', '.xyz', '.xyzn', '.xyzrgb', '.pts']:
        pcd = o3d.io.read_point_cloud(path)
    elif ext in ['.stl', '.obj', '.off']:
        mesh = o3d.io.read_triangle_mesh(path)
        if not mesh.has_triangles():
            raise RuntimeError("Mesh không hợp lệ hoặc rỗng.")
        mesh.compute_vertex_normals()
        # Uniform sampling (Poisson-disk nếu muốn mịn hơn)
        number_of_points = 200000  # điều chỉnh theo kích thước đối tượng
        pcd = mesh.sample_points_poisson_disk(number_of_points=number_of_points)
    else:
        raise RuntimeError(f"Định dạng không hỗ trợ: {ext}")

    if voxel is not None and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    return pcd

def ensure_normals(pcd, radius, max_nn=60, camera_loc=None):
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    # Hướng normal về phía camera (nếu biết)
    if camera_loc is not None:
        pcd.orient_normals_towards_camera_location(camera_loc)
    return pcd

def compute_fpfh(pcd, radius_normal, radius_feature, max_nn=100):
    pcd = ensure_normals(pcd, radius=radius_normal, max_nn=60)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn)
    )
    return fpfh

# ========== Coarse alignment ==========

def ransac_coarse(source_down, target_down, f_src, f_tgt, distance_threshold, seed=None):
    """
    Global registration bằng RANSAC Open3D.
    Trả về dict {'T', 'fitness', 'inlier_rmse', 'score'}.
    """
    if seed is not None:
        np.random.seed(seed)
        o3d.utility.random.seed(seed)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, f_src, f_tgt,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 500)
    )
    T = result.transformation
    # Heuristic score: nhiều inlier + RMSE nhỏ
    score = result.fitness / (result.inlier_rmse + 1e-6)
    return {
        'T': T.copy(),
        'fitness': result.fitness,
        'inlier_rmse': result.inlier_rmse,
        'score': score
    }

def generate_multi_hypotheses_ransac(source_down, target_down, f_src, f_tgt,
                                     base_thresh, n_runs=40, jitter_ratio=0.4, seed0=0):
    hyps = []
    for i in range(n_runs):
        thr = base_thresh * (1.0 + jitter_ratio*(np.random.rand()*2.0 - 1.0))
        h = ransac_coarse(source_down, target_down, f_src, f_tgt, distance_threshold=max(1e-6, thr), seed=seed0+i)
        hyps.append(h)
    return hyps

# ========== ICP refine & scoring có nhận thức hiển thị (visibility-aware) ==========

def icp_refine(source, target, T_init, max_corr, robust=True, max_iter=50):
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    if robust:
        loss = o3d.pipelines.registration.RobustKernel(o3d.pipelines.registration.RobustKernelType.Huber, 1.0)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_corr, T_init, estimation,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    return result.transformation, result.fitness, result.inlier_rmse

def hidden_point_visible_subset(pcd, camera_center, radius_scale=1.2):
    """
    Lấy subset điểm nhìn thấy từ camera_center qua HiddenPointRemoval.
    radius được ước từ bounding sphere của cloud.
    """
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        return pcd
    center = pts.mean(axis=0)
    rad = np.linalg.norm(pts - center, axis=1).max()
    radius = rad * radius_scale
    _, pt_map = pcd.hidden_point_removal(camera_center, radius)
    return pcd.select_by_index(pt_map)

def truncated_chamfer_one_sided(src_pts, tgt_kd, trunc):
    """
    Chamfer một phía: trung bình khoảng cách từ src -> tgt, cắt ngưỡng 'trunc'.
    """
    if src_pts.shape[0] == 0:
        return 1e6
    dists = []
    for p in src_pts:
        [_, idx, _] = tgt_kd.search_knn_vector_3d(p, 1)
        q = np.asarray(tgt_kd.geometry.points)[idx[0]]
        d = np.linalg.norm(p - q)
        dists.append(min(d, trunc))
    return float(np.mean(dists)) if len(dists) else 1e6

def visibility_aware_score(source_model, target_scene, T, camera_center, trunc=0.010):
    """
    Chấm điểm: chỉ xét phần model nhìn thấy từ camera, rồi Chamfer một phía về scene.
    Điểm 'nhỏ hơn' thì tốt hơn (RMSE).
    """
    src = copy.deepcopy(source_model).transform(T)
    vis = hidden_point_visible_subset(src, camera_center)
    tgt_kd = o3d.geometry.KDTreeFlann(target_scene)
    val = truncated_chamfer_one_sided(np.asarray(vis.points), tgt_kd, trunc=trunc)
    return val, vis

# ========== Main pipeline ==========

def main():
    ap = argparse.ArgumentParser(description="Symmetry-aware 6D pose estimation (Open3D)")
    ap.add_argument("--model", required=True, help="Đường dẫn CAD: .stl/.obj/.off hoặc point cloud .pcd/.ply/.xyz")
    ap.add_argument("--scene", required=True, help="Đường dẫn scene point cloud: .pcd/.ply/.xyz")
    ap.add_argument("--voxel", type=float, default=0.003, help="Voxel downsample (m)")
    ap.add_argument("--voxel_scene", type=float, default=None, help="Voxel scene (m), mặc định = voxel")
    ap.add_argument("--normal_radius", type=float, default=0.01, help="Bán kính estimate normal (m)")
    ap.add_argument("--feat_radius", type=float, default=0.025, help="Bán kính FPFH (m)")
    ap.add_argument("--ransac_thresh", type=float, default=0.01, help="Ngưỡng inlier cho RANSAC (m)")
    ap.add_argument("--ransac_runs", type=int, default=48, help="Số lần chạy RANSAC (multi-hypotheses)")
    ap.add_argument("--sym_n", type=int, default=4, help="Bậc đối xứng quay Cn (vd 4,6,8). Đặt 1 nếu không đối xứng.")
    ap.add_argument("--sym_axis", type=float, nargs=3, default=[0,0,1], help="Trục đối xứng (mặc định Z)")
    ap.add_argument("--dedup_eps_deg", type=float, default=3.0, help="Ngưỡng tương đương quay (độ)")
    ap.add_argument("--dedup_tau", type=float, default=0.003, help="Ngưỡng tương đương t (m)")
    ap.add_argument("--keep_topk", type=int, default=10, help="Giữ tối đa N giả thuyết sau dedup")
    ap.add_argument("--icp_corr", type=float, default=0.01, help="Max correspondence cho ICP (m)")
    ap.add_argument("--icp_iter", type=int, default=60, help="Số vòng ICP tối đa")
    ap.add_argument("--cam", type=float, nargs=3, default=[0,0,0], help="Vị trí camera (m) cho HiddenPointRemoval")
    ap.add_argument("--trunc", type=float, default=0.01, help="Truncated distance (m) cho scoring")
    ap.add_argument("--export_all", action="store_true", help="Xuất tất cả giả thuyết (PLY)")
    ap.add_argument("--out_dir", default="pose_out", help="Thư mục xuất kết quả")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    voxel_scene = args.voxel_scene if args.voxel_scene is not None else args.voxel

    print("==> Load & preprocess ...")
    model = load_point_cloud_any(args.model, voxel=args.voxel)
    scene = load_point_cloud_any(args.scene, voxel=voxel_scene)

    # Ước tính normal (hướng về camera nếu biết)
    cam_loc = np.array(args.cam, dtype=np.float64)
    model = ensure_normals(model, radius=args.normal_radius, camera_loc=cam_loc)
    scene = ensure_normals(scene, radius=args.normal_radius, camera_loc=cam_loc)

    print(f"Model pts: {np.asarray(model.points).shape[0]} | Scene pts: {np.asarray(scene.points).shape[0]}")

    print("==> Compute FPFH ...")
    f_model = compute_fpfh(model, radius_normal=args.normal_radius, radius_feature=args.feat_radius)
    f_scene = compute_fpfh(scene, radius_normal=args.normal_radius, radius_feature=args.feat_radius)

    print("==> Coarse multi-hypotheses (RANSAC) ...")
    hyps = generate_multi_hypotheses_ransac(
        model, scene, f_model, f_scene,
        base_thresh=args.ransac_thresh,
        n_runs=args.ransac_runs,
        jitter_ratio=0.4, seed0=0
    )

    # Dedup theo symmetry
    G = make_Cn_rotations(axis=np.array(args.sym_axis, dtype=np.float64), n=max(1, args.sym_n))
    hyps_T = [{'T': h['T'], 'score': h['score']} for h in hyps]
    hyps_dedup = deduplicate_hypotheses(
        hyps_T, G, eps_deg=args.dedup_eps_deg, tau=args.dedup_tau, topk=args.keep_topk
    )
    print(f"Giữ lại {len(hyps_dedup)} giả thuyết sau dedup symmetry (Cn, n={args.sym_n}).")

    # ICP refine + visibility-aware scoring
    print("==> ICP refine + Visibility-aware scoring ...")
    refined = []
    for i, h in enumerate(hyps_dedup):
        T0 = h['T']
        T_icp, fit, rmse = icp_refine(model, scene, T0, max_corr=args.icp_corr, robust=True, max_iter=args.icp_iter)
        vis_rmse, vis_cloud = visibility_aware_score(model, scene, T_icp, camera_center=cam_loc, trunc=args.trunc)

        refined.append({
            'idx': i,
            'T': T_icp,
            'fitness': float(fit),
            'rmse': float(rmse),
            'vis_rmse': float(vis_rmse),
        })
        print(f"[{i:02d}] ICP: fitness={fit:.3f}, rmse={rmse:.4f}, vis_rmse={vis_rmse:.4f}")

        if args.export_all:
            model_aligned = copy.deepcopy(model).transform(T_icp)
            o3d.io.write_point_cloud(os.path.join(args.out_dir, f"model_aligned_{i:02d}.ply"), model_aligned)
            o3d.io.write_point_cloud(os.path.join(args.out_dir, f"model_visible_{i:02d}.ply"), vis_cloud)

    # Chọn kết quả tốt nhất theo visibility RMSE (nhỏ là tốt)
    best = sorted(refined, key=lambda x: x['vis_rmse'])[0]
    T_best = best['T']
    print("\n==> BEST POSE (theo visibility-aware RMSE):")
    print(T_best)

    # Xuất kết quả
    model_best = copy.deepcopy(model).transform(T_best)
    o3d.io.write_point_cloud(os.path.join(args.out_dir, "model_aligned_best.ply"), model_best)

    # Lưu ma trận transform
    np.save(os.path.join(args.out_dir, "T_best.npy"), T_best)
    with open(os.path.join(args.out_dir, "T_best.json"), "w") as f:
        json.dump({'T': T_best.tolist(), 'note': 'Pose từ model sang scene (model* = T * model)'}, f, indent=2)

    # Lưu toàn bộ ứng viên (tuỳ chọn)
    with open(os.path.join(args.out_dir, "all_candidates.json"), "w") as f:
        json.dump(refined, f, indent=2, default=float)

    print(f"\nKết quả đã lưu trong: {args.out_dir}")
    print(" - model_aligned_best.ply")
    print(" - T_best.npy / T_best.json")
    if args.export_all:
        print(" - model_aligned_XX.ply & model_visible_XX.ply cho từng giả thuyết")

if __name__ == "__main__":
    main()

# python regis1.py --model data/K41144.stl --scene view/view_00.pcd --voxel 0.003 --normal_radius 0.01 --feat_radius 0.025 --ransac_thresh 0.01 --ransac_runs 64 --sym_n 6 --sym_axis 0 0 1 --icp_corr 0.01 --icp_iter 100 --cam 0 0 0 --export_all --out_dir pose_out
