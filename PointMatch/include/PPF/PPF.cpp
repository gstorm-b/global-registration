#include "PPF.h"

#include <fstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <sys/stat.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
// #include <pcl/filters/hidden_point_removal.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/console/print.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/filters/farthest_point_sampling.h>

#include <Eigen/Dense>
#include <memory>
#include <limits>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <vector>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>

typedef pcl::PointXYZ                              PointT;
typedef pcl::PointCloud<PointT>                    CloudT;
typedef pcl::PointNormal                           PointN;
typedef pcl::PointCloud<PointN>                    CloudN;
typedef pcl::FPFHSignature33                       FPFH33;
typedef pcl::PointCloud<FPFH33>                    FPFHCloud;

// ====== Utils ======
static CloudT::Ptr RemoveNaN(const CloudT::ConstPtr& in) {
  CloudT::Ptr out(new CloudT);
  std::vector<int> idx;
  pcl::removeNaNFromPointCloud(*in, *out, idx);
  return out;
}

static CloudT::Ptr VoxelDownsample(const CloudT::Ptr& in, float leaf) {
  CloudT::Ptr out(new CloudT);
  pcl::VoxelGrid<PointT> vg;
  vg.setInputCloud(in);
  vg.setLeafSize(leaf, leaf, leaf);
  vg.setMinimumPointsNumberPerVoxel(1);
  vg.filter(*out);
  return out;
}

static CloudN::Ptr CopyXYZToPointNormal(const CloudT::Ptr& in) {
  CloudN::Ptr out(new CloudN);
  out->resize(in->size());
  for (std::size_t i = 0; i < in->size(); ++i) {
    (*out)[i].x = (*in)[i].x;
    (*out)[i].y = (*in)[i].y;
    (*out)[i].z = (*in)[i].z;
    // normal sẽ được compute sau
  }
  return out;
}

static CloudN::Ptr EstimateNormals(const CloudT::Ptr& xyz, float radius) {
  CloudN::Ptr pn = CopyXYZToPointNormal(xyz);

  pcl::NormalEstimationOMP<PointT, PointN> ne;
  ne.setInputCloud(xyz);

  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(radius);
  ne.compute(*pn);

  return pn;
}

static FPFHCloud::Ptr ComputeFPFH(const CloudT::Ptr& xyz,
                                  const CloudN::Ptr& pn,
                                  float radius) {
  FPFHCloud::Ptr feat(new FPFHCloud);

  pcl::FPFHEstimationOMP<PointT, PointN, FPFH33> est;
  est.setInputCloud(xyz);
  est.setInputNormals(pn);

  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
  est.setSearchMethod(tree);
  est.setRadiusSearch(radius);
  est.compute(*feat);

  return feat;
}

static void ApplyTransformXYZ(const CloudT::ConstPtr& in,
                              const Eigen::Matrix4f& T,
                              CloudT::Ptr& out) {
  out->clear();
  out->reserve(in->size());
  for (std::size_t i = 0; i < in->size(); ++i) {
    const PointT& pt = (*in)[i];
    Eigen::Vector4f v(pt.x, pt.y, pt.z, 1.0f);
    Eigen::Vector4f w = T * v;
    PointT q;
    q.x = w.x(); q.y = w.y(); q.z = w.z();
    out->push_back(q);
  }
}

// ====== API ======
PoseEstimationParams::PoseEstimationParams()
: voxel_size(0.001f),
  normal_radius(0.010f),
  feature_radius(0.025f),
  sac_max_iters(50000),
  number_of_sample(3),
  sac_corr_randomness(5),
  sac_sim_threshold(0.90f),
  sac_max_corr_dist(0.030f),
  sac_inlier_fraction(0.25f),
  icp_max_corr_dist(0.020f),
  icp_max_iters(50),
  icp_trans_eps(1e-6f),
  icp_fit_eps(1e-6f),
  T_init(Eigen::Matrix4f::Identity()),
  use_external_init(false)
{}

Eigen::Matrix4f EstimatePoseRobustPCL(
  const CloudT::ConstPtr& object_in,
  const CloudT::ConstPtr& scene_in,
  const PoseEstimationParams& params,
  CloudT::Ptr* out_coarse_aligned)
{
  // 0) Dọn NaN
  // CloudT::Ptr obj_clean = RemoveNaN(object_in);
  // CloudT::Ptr scn_clean = RemoveNaN(scene_in);

  // 1) Downsample
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
  CloudT::Ptr scn_ds(new CloudT);
  pcl::copyPointCloud(*scene_in, *scn_ds);

  CloudT::Ptr obj_clean(new CloudT);
  pcl::copyPointCloud(*object_in, *obj_clean);
  CloudT::Ptr obj_ds = VoxelDownsample(obj_clean, params.voxel_size);
  // CloudT::Ptr scn_ds = VoxelDownsample(*scene_in, params.voxel_size);

  // 2) Normals
  CloudN::Ptr obj_n = EstimateNormals(obj_ds, params.normal_radius);
  CloudN::Ptr scn_n = EstimateNormals(scn_ds, params.normal_radius);

  // 3) FPFH
  FPFHCloud::Ptr obj_fpfh = ComputeFPFH(obj_ds, obj_n, params.feature_radius);
  FPFHCloud::Ptr scn_fpfh = ComputeFPFH(scn_ds, scn_n, params.feature_radius);

  // 4) Coarse: RANSAC prerejective
  pcl::SampleConsensusPrerejective<PointN, PointN, FPFH33> align;
  align.setInputSource(obj_n);
  align.setSourceFeatures(obj_fpfh);
  align.setInputTarget(scn_n);
  align.setTargetFeatures(scn_fpfh);

  align.setMaximumIterations(params.sac_max_iters);
  align.setNumberOfSamples(params.number_of_sample);
  align.setCorrespondenceRandomness(params.sac_corr_randomness);
  align.setSimilarityThreshold(params.sac_sim_threshold);
  align.setMaxCorrespondenceDistance(params.sac_max_corr_dist);
  align.setInlierFraction(params.sac_inlier_fraction);

  CloudN::Ptr obj_coarse(new CloudN);
  if (params.use_external_init) {
    align.align(*obj_coarse, params.T_init);
  } else {
    align.align(*obj_coarse);
  }

  if (!align.hasConverged()) {
    // thất bại -> trả Identity, out_coarse_aligned rỗng
    return Eigen::Matrix4f::Identity();
  }

  Eigen::Matrix4f T_coarse = align.getFinalTransformation();

  // 5) nếu cần, trả thêm bản aligned theo T_coarse nhưng ở độ phân giải gốc
  if (out_coarse_aligned != NULL) {
    if (*out_coarse_aligned == NULL) {
      *out_coarse_aligned = CloudT::Ptr(new CloudT);
    }
    ApplyTransformXYZ(object_in, T_coarse, *out_coarse_aligned);
  }
  return T_coarse;
}

// ================== Utils cơ bản ==================
// static CloudT::Ptr RemoveNaN(const CloudT::ConstPtr& in) {
//   CloudT::Ptr out(new CloudT);
//   std::vector<int> idx;
//   pcl::removeNaNFromPointCloud(*in, *out, idx);
//   return out;
// }

// static CloudT::Ptr VoxelDownsample(const CloudT::Ptr& in, float leaf) {
//   CloudT::Ptr out(new CloudT);
//   pcl::VoxelGrid<PointT> vg;
//   vg.setInputCloud(in);
//   vg.setLeafSize(leaf, leaf, leaf);
//   vg.setMinimumPointsNumberPerVoxel(1);
//   vg.filter(*out);
//   return out;
// }

// static CloudN::Ptr CopyXYZToPointNormal(const CloudT::Ptr& in) {
//   CloudN::Ptr out(new CloudN);
//   out->resize(in->size());
//   for (std::size_t i = 0; i < in->size(); ++i) {
//     (*out)[i].x = (*in)[i].x;
//     (*out)[i].y = (*in)[i].y;
//     (*out)[i].z = (*in)[i].z;
//   }
//   return out;
// }

// static CloudN::Ptr EstimateNormals(const CloudT::Ptr& xyz, float radius) {
//   CloudN::Ptr pn = CopyXYZToPointNormal(xyz);
//   pcl::NormalEstimationOMP<PointT, PointN> ne;
//   ne.setInputCloud(xyz);
//   pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
//   ne.setSearchMethod(tree);
//   ne.setRadiusSearch(radius);
//   ne.compute(*pn);
//   return pn;
// }

// static FPFHCloud::Ptr ComputeFPFH(const CloudT::Ptr& xyz,
//                                   const CloudN::Ptr& pn,
//                                   float radius) {
//   FPFHCloud::Ptr feat(new FPFHCloud);
//   pcl::FPFHEstimationOMP<PointT, PointN, FPFH33> est;
//   est.setInputCloud(xyz);
//   est.setInputNormals(pn);
//   pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
//   est.setSearchMethod(tree);
//   est.setRadiusSearch(radius);
//   est.compute(*feat);
//   return feat;
// }

// static void ApplyTransformXYZ(const CloudT::ConstPtr& in,
//                               const Eigen::Matrix4f& T,
//                               CloudT::Ptr& out) {
//   if (!out) out = CloudT::Ptr(new CloudT);
//   out->clear();
//   out->reserve(in->size());
//   for (std::size_t i = 0; i < in->size(); ++i) {
//     const PointT& pt = (*in)[i];
//     Eigen::Vector4f v(pt.x, pt.y, pt.z, 1.0f);
//     Eigen::Vector4f w = T * v;
//     PointT q; q.x = w.x(); q.y = w.y(); q.z = w.z();
//     out->push_back(q);
//   }
// }

static float RotationAngleDeg(const Eigen::Matrix4f& A,
                              const Eigen::Matrix4f& B) {
  Eigen::Matrix3f R = A.block<3,3>(0,0).transpose() * B.block<3,3>(0,0);
  float tr = (R.trace() - 1.0f) * 0.5f;
  if (tr > 1.0f) tr = 1.0f; if (tr < -1.0f) tr = -1.0f;
  float angle = std::acos(tr);
  return angle * 180.0f / static_cast<float>(M_PI);
}

static float TranslationDist(const Eigen::Matrix4f& A,
                             const Eigen::Matrix4f& B) {
  Eigen::Vector3f ta(A(0,3), A(1,3), A(2,3));
  Eigen::Vector3f tb(B(0,3), B(1,3), B(2,3));
  return (ta - tb).norm();
}

static bool IsSimilarPose(const Eigen::Matrix4f& A,
                          const Eigen::Matrix4f& B,
                          float trans_eps, float rot_deg) {
  return (TranslationDist(A,B) <= trans_eps) &&
         (RotationAngleDeg(A,B) <= rot_deg);
}

// Đếm inlier sau khi áp T: dùng KDTree trên scene_ds.
static int CountInliers(const CloudT::Ptr& obj_ds,
                        const CloudT::Ptr& scn_ds,
                        float inlier_dist,
                        const Eigen::Matrix4f& T) {
  pcl::KdTreeFLANN<PointT> kdtree;
  kdtree.setInputCloud(scn_ds);

  int inliers = 0;
  std::vector<int> idx(1);
  std::vector<float> d2(1);

  for (std::size_t i = 0; i < obj_ds->size(); ++i) {
    const PointT& p = (*obj_ds)[i];
    Eigen::Vector4f v(p.x, p.y, p.z, 1.0f);
    Eigen::Vector4f w = T * v;
    PointT q; q.x = w.x(); q.y = w.y(); q.z = w.z();
    int found = kdtree.nearestKSearch(q, 1, idx, d2);
    if (found > 0) {
      float dist = std::sqrt(d2[0]);
      if (dist <= inlier_dist) ++inliers;
    }
  }
  return inliers;
}

// Subsample ngẫu nhiên chỉ số
static void RandomSubsampleIndices(std::size_t N,
                                   float ratio,
                                   std::vector<int>& out_idx,
                                   std::mt19937& rng) {
  out_idx.clear();
  if (ratio >= 1.0f) {
    out_idx.reserve(N);
    for (std::size_t i = 0; i < N; ++i) out_idx.push_back(static_cast<int>(i));
    return;
  }
  std::vector<int> all;
  all.reserve(N);
  for (std::size_t i = 0; i < N; ++i) all.push_back(static_cast<int>(i));
  std::shuffle(all.begin(), all.end(), rng);
  std::size_t M = static_cast<std::size_t>(std::max(16.0f, std::floor(ratio * static_cast<float>(N))));
  if (M > all.size()) M = all.size();
  out_idx.assign(all.begin(), all.begin() + M);
}

template<typename CloudType>
static typename CloudType::Ptr GatherByIndex(const typename CloudType::Ptr& in,
                                             const std::vector<int>& indices) {
  typename CloudType::Ptr out(new CloudType);
  out->reserve(indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    out->push_back((*in)[indices[i]]);
  }
  return out;
}

// ================== Constructors ==================
// PoseEstimationParams::PoseEstimationParams()
// : voxel_size(0.005f),
//   normal_radius(0.010f),
//   feature_radius(0.025f),
//   sac_max_iters(50000),
//   sac_corr_randomness(5),
//   sac_sim_threshold(0.90f),
//   sac_max_corr_dist(0.030f),
//   sac_inlier_fraction(0.25f),
//   icp_max_corr_dist(0.020f),
//   icp_max_iters(50),
//   icp_trans_eps(1e-6f),
//   icp_fit_eps(1e-6f),
//   T_init(Eigen::Matrix4f::Identity()),
//   use_external_init(false)
// {}

MultiHypothesisParams::MultiHypothesisParams()
: num_hypotheses(32),
  refine_top_k(5),
  cluster_trans_eps(0.010f),
  cluster_rot_deg(5.0f),
  eval_inlier_dist_coarse(0.030f),
  eval_inlier_dist_refine(0.020f),
  source_subsample_ratio(0.8f),
  random_seed(42),
  use_fixed_seed(false)
{}

// ================== Coarse 1 lần (tái dùng) ==================
Eigen::Matrix4f EstimatePoseCoarsePCL(
  const CloudT::ConstPtr& object_in,
  const CloudT::ConstPtr& scene_in,
  const PoseEstimationParams& params,
  CloudT::Ptr* out_coarse_aligned)
{
  // CloudT::Ptr obj_clean = RemoveNaN(object_in);
  // CloudT::Ptr scn_clean = RemoveNaN(scene_in);

  // CloudT::Ptr obj_ds = VoxelDownsample(obj_clean, params.voxel_size);
  // CloudT::Ptr scn_ds = VoxelDownsample(scn_clean, params.voxel_size);

  CloudT::Ptr scn_ds(new CloudT);
  pcl::copyPointCloud(*scene_in, *scn_ds);

  CloudT::Ptr obj_clean(new CloudT);
  pcl::copyPointCloud(*object_in, *obj_clean);
  CloudT::Ptr obj_ds = VoxelDownsample(obj_clean, params.voxel_size);

  CloudN::Ptr obj_n = EstimateNormals(obj_ds, params.normal_radius);
  CloudN::Ptr scn_n = EstimateNormals(scn_ds, params.normal_radius);

  FPFHCloud::Ptr obj_fpfh = ComputeFPFH(obj_ds, obj_n, params.feature_radius);
  FPFHCloud::Ptr scn_fpfh = ComputeFPFH(scn_ds, scn_n, params.feature_radius);

  pcl::SampleConsensusPrerejective<PointN, PointN, FPFH33> align;
  align.setInputSource(obj_n);
  align.setSourceFeatures(obj_fpfh);
  align.setInputTarget(scn_n);
  align.setTargetFeatures(scn_fpfh);
  align.setMaximumIterations(params.sac_max_iters);
  align.setNumberOfSamples(3);
  align.setCorrespondenceRandomness(params.sac_corr_randomness);
  align.setSimilarityThreshold(params.sac_sim_threshold);
  align.setMaxCorrespondenceDistance(params.sac_max_corr_dist);
  align.setInlierFraction(params.sac_inlier_fraction);

  CloudN::Ptr obj_coarse(new CloudN);
  if (params.use_external_init) align.align(*obj_coarse, params.T_init);
  else                          align.align(*obj_coarse);

  if (!align.hasConverged()) return Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T_coarse = align.getFinalTransformation();

  if (out_coarse_aligned != NULL) {
    if (*out_coarse_aligned == NULL) *out_coarse_aligned = CloudT::Ptr(new CloudT);
    ApplyTransformXYZ(object_in, T_coarse, *out_coarse_aligned);
  }
  return T_coarse;
}

// ================== Multi-hypothesis ==================
Eigen::Matrix4f EstimatePoseMultiHypothesisPCL(
  const CloudT::ConstPtr& object_in,
  const CloudT::ConstPtr& scene_in,
  const PoseEstimationParams& base_params,
  const MultiHypothesisParams& mh_params,
  float* out_best_fitness,
  CloudT::Ptr* out_aligned,
  std::vector<PoseHypothesis>* out_all_hypotheses)
{
  if (out_best_fitness != NULL) {
    *out_best_fitness = std::numeric_limits<float>::infinity();
  }

  CloudT::Ptr scn_ds(new CloudT);
  pcl::copyPointCloud(*scene_in, *scn_ds);

  CloudT::Ptr obj_clean(new CloudT);
  pcl::copyPointCloud(*object_in, *obj_clean);
  CloudT::Ptr obj_ds = VoxelDownsample(obj_clean, base_params.voxel_size);

  // 0) Tiền xử lý & feature TÍNH MỘT LẦN
  // CloudT::Ptr obj_clean = RemoveNaN(object_in);
  // CloudT::Ptr scn_clean = RemoveNaN(scene_in);

  // CloudT::Ptr obj_ds = VoxelDownsample(obj_clean, base_params.voxel_size);
  // CloudT::Ptr scn_ds = VoxelDownsample(scn_clean, base_params.voxel_size);

  CloudN::Ptr obj_n = EstimateNormals(obj_ds, base_params.normal_radius);
  CloudN::Ptr scn_n = EstimateNormals(scn_ds, base_params.normal_radius);

  FPFHCloud::Ptr obj_fpfh = ComputeFPFH(obj_ds, obj_n, base_params.feature_radius);
  FPFHCloud::Ptr scn_fpfh = ComputeFPFH(scn_ds, scn_n, base_params.feature_radius);

  // KDTree cho chấm coarse/refine
  int total_src = static_cast<int>(obj_ds->size());
  int total_tgt = static_cast<int>(scn_ds->size());
  if (total_src < 20 || total_tgt < 20) {
    // quá ít điểm -> khó ước lượng
    return Eigen::Matrix4f::Identity();
  }

  // RNG
  std::mt19937 rng;
  if (mh_params.use_fixed_seed) rng.seed(mh_params.random_seed);
  else                          rng.seed(static_cast<unsigned int>(std::random_device{}()));

  // 1) Sinh nhiều coarse pose
  std::vector<PoseHypothesis> cand;
  cand.reserve(mh_params.num_hypotheses);

  for (int h = 0; h < mh_params.num_hypotheses; ++h) {
    // random subsample nguồn để thay đổi phân bố mẫu RANSAC
    std::vector<int> idx_src;
    RandomSubsampleIndices(obj_ds->size(), mh_params.source_subsample_ratio, idx_src, rng);
    CloudT::Ptr   obj_ds_sub = GatherByIndex<CloudT>(obj_ds, idx_src);
    CloudN::Ptr   obj_n_sub  = GatherByIndex<CloudN>(obj_n,  idx_src);
    FPFHCloud::Ptr obj_fpfh_sub = GatherByIndex<FPFHCloud>(obj_fpfh, idx_src);

    // RANSAC prerejective
    pcl::SampleConsensusPrerejective<PointN, PointN, FPFH33> align;
    align.setInputSource(obj_n_sub);
    align.setSourceFeatures(obj_fpfh_sub);
    align.setInputTarget(scn_n);
    align.setTargetFeatures(scn_fpfh);
    align.setMaximumIterations(base_params.sac_max_iters);
    align.setNumberOfSamples(3);
    align.setCorrespondenceRandomness(base_params.sac_corr_randomness);
    align.setSimilarityThreshold(base_params.sac_sim_threshold);
    align.setMaxCorrespondenceDistance(base_params.sac_max_corr_dist);
    align.setInlierFraction(base_params.sac_inlier_fraction);

    CloudN::Ptr obj_coarse(new CloudN);
    if (base_params.use_external_init) align.align(*obj_coarse, base_params.T_init);
    else                               align.align(*obj_coarse);

    if (!align.hasConverged()) {
      continue; // bỏ qua lần này
    }
    Eigen::Matrix4f T_coarse = align.getFinalTransformation();

    // chấm coarse: tỉ lệ inlier trên obj_ds (đầy đủ) để công bằng
    int in_coarse = CountInliers(obj_ds, scn_ds, mh_params.eval_inlier_dist_coarse, T_coarse);
    float ratio = static_cast<float>(in_coarse) / static_cast<float>(obj_ds->size());

    PoseHypothesis ph;
    ph.T = T_coarse;
    ph.coarse_inliers = in_coarse;
    ph.coarse_inlier_ratio = ratio;
    cand.push_back(ph);
  }

  if (cand.empty()) {
    // không có coarse hợp lệ
    return Eigen::Matrix4f::Identity();
  }

  // 2) Gom cụm pose (loại trùng gần nhau), giữ đại diện điểm số cao nhất
  std::vector<PoseHypothesis> reps; // representatives
  for (std::size_t i = 0; i < cand.size(); ++i) {
    bool merged = false;
    for (std::size_t j = 0; j < reps.size(); ++j) {
      if (IsSimilarPose(cand[i].T, reps[j].T, mh_params.cluster_trans_eps, mh_params.cluster_rot_deg)) {
        // giữ pose có coarse_inlier_ratio cao hơn
        if (cand[i].coarse_inlier_ratio > reps[j].coarse_inlier_ratio) {
          reps[j] = cand[i];
        }
        merged = true;
        break;
      }
    }
    if (!merged) reps.push_back(cand[i]);
  }

  // 3) Sắp xếp theo điểm coarse, lấy top-K để refine
  std::sort(reps.begin(), reps.end(),
            [](const PoseHypothesis& a, const PoseHypothesis& b){
              return a.coarse_inlier_ratio > b.coarse_inlier_ratio;
            });
  int K = std::min(mh_params.refine_top_k, static_cast<int>(reps.size()));

  // 4) Refine bằng ICP point-to-plane
  pcl::IterativeClosestPointWithNormals<PointN, PointN> icp;
  icp.setInputTarget(scn_n);
  icp.setMaximumIterations(base_params.icp_max_iters);
  icp.setMaxCorrespondenceDistance(base_params.icp_max_corr_dist);
  icp.setTransformationEpsilon(base_params.icp_trans_eps);
  icp.setEuclideanFitnessEpsilon(base_params.icp_fit_eps);
  pcl::registration::TransformationEstimationPointToPlaneLLS<PointN, PointN>::Ptr te(
      new pcl::registration::TransformationEstimationPointToPlaneLLS<PointN, PointN>());
  icp.setTransformationEstimation(te);

  Eigen::Matrix4f T_best = reps[0].T; // fallback: best coarse
  float best_fitness = std::numeric_limits<float>::infinity();
  int best_ref_inliers = -1;
  int best_idx = -1;

  for (int i = 0; i < K; ++i) {
    // ICP với initial guess = T_coarse, source = obj_n
    CloudN::Ptr obj_nn(new CloudN);
    HPRN(obj_n, {0, 0, 0}, 2, obj_nn); //calculate from i view with 

    CloudN::Ptr src_refined(new CloudN);
    icp.setInputSource(obj_nn);
    icp.align(*src_refined, reps[i].T);

    if (icp.hasConverged()) {
      Eigen::Matrix4f T_ref = icp.getFinalTransformation();
      float fitness = static_cast<float>(icp.getFitnessScore());
      int in_ref = CountInliers(obj_ds, scn_ds, mh_params.eval_inlier_dist_refine, T_ref);

      reps[i].refined = true;
      reps[i].icp_fitness = fitness;
      reps[i].refine_inliers = in_ref;

      // chọn theo fitness, hoà giải bằng inlier refine cao hơn
      bool better = false;
      if (fitness < best_fitness) better = true;
      else if (fitness == best_fitness && in_ref > best_ref_inliers) better = true;

      if (better) {
        best_fitness = fitness;
        best_ref_inliers = in_ref;
        T_best = T_ref;
        best_idx = i;
      }
    }
  }

  // Nếu K=0 hoặc ICP không hội tụ pose nào -> dùng coarse tốt nhất
  if (best_idx < 0) {
    T_best = reps[0].T;
    best_fitness = std::numeric_limits<float>::infinity();
  }

  // Xuất aligned nếu cần
  if (out_aligned != NULL) {
    if (*out_aligned == NULL) *out_aligned = CloudT::Ptr(new CloudT);
    ApplyTransformXYZ(object_in, T_best, *out_aligned);
  }
  if (out_best_fitness != NULL) *out_best_fitness = best_fitness;

  // gom toàn bộ ứng viên nếu yêu cầu (coarse + refine cập nhật)
  if (out_all_hypotheses != NULL) {
    // ghép reps (đã cluster) trước, nhưng nếu muốn đầy đủ cả cand, có thể trả cand.
    *out_all_hypotheses = reps;
  }

  return T_best;
}

bool LoadParamsFromJSON(const ConfigReader& cfg, PoseEstimationParams& params) {
  params.voxel_size        = cfg.get<float>("voxel_size", params.voxel_size);
  params.normal_radius     = cfg.get<float>("normal_radius", params.normal_radius);
  params.feature_radius    = cfg.get<float>("feature_radius", params.feature_radius);

  params.sac_max_iters     = cfg.get<int>("sac_max_iters", params.sac_max_iters);
  params.number_of_sample     = cfg.get<int>("number_of_sample", params.number_of_sample);
  params.sac_corr_randomness = cfg.get<int>("sac_corr_randomness", params.sac_corr_randomness);
  params.sac_sim_threshold = cfg.get<float>("sac_sim_threshold", params.sac_sim_threshold);
  params.sac_max_corr_dist = cfg.get<float>("sac_max_corr_dist", params.sac_max_corr_dist);
  params.sac_inlier_fraction = cfg.get<float>("sac_inlier_fraction", params.sac_inlier_fraction);

  params.icp_max_corr_dist = cfg.get<float>("icp_max_corr_dist", params.icp_max_corr_dist);
  params.icp_max_iters     = cfg.get<int>("icp_max_iters", params.icp_max_iters);
  params.icp_trans_eps     = cfg.get<float>("icp_trans_eps", params.icp_trans_eps);
  params.icp_fit_eps       = cfg.get<float>("icp_fit_eps", params.icp_fit_eps);
  
  return true;
}

DescriptorPPF::DescriptorPPF() {
	customViewer.init();
}

void DescriptorPPF::setModelPath(std::string model_path) {
	this->model_dir = model_path;
}

void DescriptorPPF::setModelPcdPath(std::string model_path) {
	this->model_pcd_dir = model_path;
}

bool DescriptorPPF::loadModel() {
	// std::cout << "Step 1: Load STL file and perform point sampling from each view" << std::endl;
	// std::getchar();
	std::string file_extension = model_dir.substr(model_dir.find_last_of('.'));
	if (file_extension == ".stl" || file_extension == ".STL") {
		std::cout << "Loading mesh..." << std::endl;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
		meshSampling(model_dir, 1000000, 0.0005f, false, cloud_xyz);
		pcl::copyPointCloud(*cloud_xyz, *model_sampling);
	}
	else if (file_extension == ".pcd") {
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
		if (pcl::io::loadPCDFile(model_dir, *cloud_xyz) < 0) {
			
			std::cout << "Error loading PCD model cloud." << std::endl;
			return false;
		}
		pcl::copyPointCloud(*cloud_xyz, *model_sampling);
	}

	//---- Calculate point cloud from 6 views and combine ------
	std::vector<std::vector<float>> camera_pos(6);
	pcl::PointXYZ minPt, maxPt, avgPt;

	pcl::getMinMax3D(*model_sampling, minPt, maxPt);
	avgPt.x = (minPt.x + maxPt.x) / 2;
	avgPt.y = (minPt.y + maxPt.y) / 2;
	avgPt.z = (minPt.z + maxPt.z) / 2;

	float cube_length = std::max(maxPt.x - minPt.x, std::max(maxPt.y - minPt.y, maxPt.z - minPt.z));

	minPt.x = avgPt.x - cube_length;
	minPt.y = avgPt.y - cube_length;
	minPt.z = avgPt.z - cube_length;
	maxPt.x = avgPt.x + cube_length;
	maxPt.y = avgPt.y + cube_length;
	maxPt.z = avgPt.z + cube_length;

	camera_pos[0] = { avgPt.x, minPt.y, avgPt.z };
	camera_pos[1] = { maxPt.x, avgPt.y, avgPt.z };
	camera_pos[2] = { avgPt.x, maxPt.y, avgPt.z };
	camera_pos[3] = { minPt.x, avgPt.y, avgPt.z };
	camera_pos[4] = { avgPt.x, avgPt.y, maxPt.z };
	camera_pos[5] = { avgPt.x, avgPt.y, minPt.z };

	for (int i = 0; i < static_cast<int>(camera_pos.size()); ++i) {
		std::cout << "Preparing Viewpoint " << i << ".....\n";
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_HPR(new pcl::PointCloud<pcl::PointXYZ>());
		HPR(model_sampling, camera_pos[i], 3, cloud_xyz_HPR); //calculate from i view with 
		
		// customViewer.viewer->removeAllShapes();
		// customViewer.viewer->removeAllPointClouds();
		// customViewer.viewer->addPointCloud(cloud_xyz_HPR);
		// std::getchar();

		if ((i==0) || (i==2)) {
			continue;
		}

    // if (i!=5) {
		// 	continue;
		// }

		*model += *cloud_xyz_HPR;
	}

	std::cout << "Assemble all viewpoints...\n";
	// customViewer.viewer->removeAllShapes();
	// customViewer.viewer->removeAllPointClouds();
	// customViewer.viewer->addPointCloud(model);
	// std::getchar();

	// ------- centering the model ----------------
	Eigen::Vector3d sum_of_pos = Eigen::Vector3d::Zero();
	for (const auto& p : *(model)) sum_of_pos += p.getVector3fMap().cast<double>();

	Eigen::Matrix4d transform_centering = Eigen::Matrix4d::Identity();
	transform_centering.topRightCorner<3, 1>() = -sum_of_pos / model->size();

	pcl::transformPointCloud(*model, *model, transform_centering);
	pcl::transformPointCloud(*model, *model, Eigen::Vector3f(0, 0, 0), Eigen::Quaternionf(0.7071, 0, -0.7071, 0));

	std::cout << "Centering Model...\n";
	customViewer.viewer->removeAllShapes();
	customViewer.viewer->removeAllPointClouds();
	customViewer.viewer->addPointCloud(model);
	// std::getchar();

	return true;
}

bool DescriptorPPF::loadModelPCD() {
	pcl::io::loadPCDFile(model_pcd_dir, *model);

	// ------- centering the model ----------------
	Eigen::Vector3d sum_of_pos = Eigen::Vector3d::Zero();
	for (const auto& p : *(model)) sum_of_pos += p.getVector3fMap().cast<double>();

	Eigen::Matrix4d transform_centering = Eigen::Matrix4d::Identity();
	transform_centering.topRightCorner<3, 1>() = -sum_of_pos / model->size();

	pcl::transformPointCloud(*model, *model, transform_centering);
	pcl::transformPointCloud(*model, *model, Eigen::Vector3f(0, 0, 0), Eigen::Quaternionf(0.7071, 0, -0.7071, 0));

	std::cout << "Centering Model...\n";
	customViewer.viewer->removeAllShapes();
	customViewer.viewer->removeAllPointClouds();
	customViewer.viewer->addPointCloud(model);
	std::getchar();

	return true;
}

void DescriptorPPF::saveToPCD(std::string path) {
	pcl::io::savePCDFileASCII(path, *model);
}

void DescriptorPPF::createSimScene(ConfigReader &cfg) {
  // Move by 10 mm in X, rotate 30 deg around Z, pivot at (0,0,0)
  float tx = 0.20f, ty = 0.50f, tz = 0.8f;
  float roll = 0.0f, pitch = 0.0f, yaw = static_cast<float>(M_PI/6.0); // 30 deg
  
  tx = cfg.get<float>("simulate_tx",  tx);
  ty = cfg.get<float>("simulate_ty",  ty);
  tz = cfg.get<float>("simulate_tz",  tz);

  roll = cfg.get<float>("simulate_roll", 0.0f);
  pitch = cfg.get<float>("simulate_pitch", 0.0f);
  yaw = cfg.get<float>("simulate_yaw", 0.0f);
  roll = static_cast<float>(M_PI/ roll);
  pitch = static_cast<float>(M_PI/ pitch);
  yaw = static_cast<float>(M_PI/ yaw);
  // roll = static_cast<float>(M_PI/ (cfg.get<float>("simulate_roll",  0.0f)));
  // pitch = static_cast<float>(M_PI/ (cfg.get<float>("simulate_pitch",  0.0f)));
  // yaw = static_cast<float>(M_PI/ (cfg.get<float>("simulate_yaw",  6.0f)));
  std::cout << "translate scene: " << tx << ", " << ty << ", " << tz << ", " << roll << ", " << pitch << ", " << yaw << "\n";  

  Eigen::Vector3f pivot(0.f, 0.f, 0.f);

	pcl::PointCloud<pcl::PointXYZ>::Ptr dst(new pcl::PointCloud<pcl::PointXYZ>());
  transformCloudRPY(*model, *dst, tx, ty, tz, roll, pitch, yaw, pivot);

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_HPR(new pcl::PointCloud<pcl::PointXYZ>());
	HPR(dst, {0, 0, 0}, 2, cloud_xyz_HPR); //calculate from i view with 

	// pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_down(new pcl::PointCloud<pcl::PointXYZ>());
	// pcl::VoxelGrid<pcl::PointXYZ> vg;
	// vg.setInputCloud(cloud_xyz_HPR);
	// vg.setLeafSize(0.001f, 0.001f, 0.001f);
	// vg.setDownsampleAllData(true);
	// vg.filter(*cloud_down);

	pcl::copyPointCloud(*cloud_xyz_HPR, *scene);

	std::cout << "Create scene...\n";
	customViewer.viewer->removeAllShapes();
	customViewer.viewer->removeAllPointClouds();
	customViewer.viewer->addPointCloud(scene);
	std::getchar();

  pcl::copyPointCloud(*scene, *scene_ori);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::FarthestPointSampling<pcl::PointXYZ> fps;
  fps.setInputCloud(scene);
  fps.setSample(cfg.get<float>("thersh_point",  1024));     // số điểm cần lấy
  fps.setSeed(42);         // tùy chọn, để lặp lại kết quả
  fps.filter(*cloud_out);   // hoặc lấy chỉ số:
  std::cout << "Apply farthest point sampling...\n";
	customViewer.viewer->removeAllShapes();
	customViewer.viewer->removeAllPointClouds();
	customViewer.viewer->addPointCloud(cloud_out);
	std::getchar();

  pcl::copyPointCloud(*cloud_out, *scene);
}

// Áp rigid transform T (4x4) lên point cloud XYZ.
// in  : cloud_src (ConstPtr), T
// out : cloud_dst (Ptr); nếu nullptr sẽ được cấp phát.
template <typename PointT>
void TransformPointCloudXYZ(const typename pcl::PointCloud<PointT>::ConstPtr& cloud_src,
                            const Eigen::Matrix4f& T,
                            typename pcl::PointCloud<PointT>::Ptr& cloud_dst) {
    if (!cloud_dst) {
        cloud_dst = typename pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
    }
    cloud_dst->clear();
    cloud_dst->reserve(cloud_src->size());

    for (std::size_t i = 0; i < cloud_src->size(); ++i) {
        const PointT& p = (*cloud_src)[i];
        Eigen::Vector4f v(p.x, p.y, p.z, 1.0f);
        Eigen::Vector4f w = T * v;

        PointT q;
        q.x = w.x(); q.y = w.y(); q.z = w.z();
        // Giữ nguyên các trường khác nếu có (rgb, intensity...) nếu PointT có:
        // Lưu ý: chỉ copy nếu PointT có các field tương ứng.
        // Ví dụ với PointXYZRGB:
        // ((pcl::PointXYZRGB&)q).r = ((const pcl::PointXYZRGB&)p).r; ...
        cloud_dst->push_back(q);
    }
}

void DescriptorPPF::match(ConfigReader &cfg) {
  PoseEstimationParams p; // dùng mặc định hợp lý
  LoadParamsFromJSON(cfg, p);

  // Có thể tinh chỉnh:
  // p.voxel_size = 0.004f; p.feature_radius = 0.024f; v.v.

  float fitness = 0.0f;
  pcl::PointCloud<pcl::PointXYZ>::Ptr coarse_aligned;
  Eigen::Matrix4f T = EstimatePoseRobustPCL(model, scene, p, &coarse_aligned);

  std::cout << "T_final =\n" << T << "\n";
  // std::cout << "fitness = " << fitness << "\n";

  pcl::PointCloud<pcl::PointXYZ>::Ptr model_in_scene;
  TransformPointCloudXYZ<pcl::PointXYZ>(model, T, model_in_scene);

  if (coarse_aligned) {
    // customViewer.viewer->removeAllShapes();
	  // customViewer.viewer->removeAllPointClouds();
    customViewer.viewer->addPointCloud(scene_ori,
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(scene_ori, 0, 0, 255), "scene");
    customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scene");
    customViewer.viewer->addPointCloud(model_in_scene,
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(model_in_scene, 0, 255, 0), "aligned");
    // pcl::io::savePCDFileBinary("aligned.pcd", *aligned);
  }

  // MultiHypothesisParams mh;
  // mh.num_hypotheses = 32;
  // mh.refine_top_k = 6;
  // mh.cluster_trans_eps = 0.008f;   // 8mm
  // mh.cluster_rot_deg   = 5.0f;
  // mh.eval_inlier_dist_coarse = 0.030f;
  // mh.eval_inlier_dist_refine = 0.015f;
  // mh.source_subsample_ratio = 0.75f;
  // mh.use_fixed_seed = true; 
  // mh.random_seed = 1234;

  // float best_fit = 0.0f;
  // pcl::PointCloud<pcl::PointXYZ>::Ptr aligned;
  // std::vector<PoseHypothesis> hypos;

  // Eigen::Matrix4f T = EstimatePoseMultiHypothesisPCL(
  //     model, scene, p, mh, &best_fit, &aligned, &hypos);
    
  // std::cout << "T_final =\n" << T << "\n";
  // // std::cout << "fitness = " << fitness << "\n";

  // pcl::PointCloud<pcl::PointXYZ>::Ptr model_in_scene;
  // TransformPointCloudXYZ<pcl::PointXYZ>(model, T, model_in_scene);

  // if (aligned) {
  //   customViewer.viewer->addPointCloud(model_in_scene,
  //   pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(model_in_scene, 0, 255, 0), "aligned");
  //   // pcl::io::savePCDFileBinary("aligned.pcd", *aligned);
  // }
	return;
}

// bool DescriptorPPF::prepareModelDescriptor()
// {
// 	std::cout << "Preparing Model Descriptor Offline.....\n" << std::endl;
// 	//  Load model cloud
// 	if (!loadModel())
// 		return false;

// 	// std::cout << "Step 2: Prepare Point Pair Feature descriptors of model\n";
// 	// //Model diameter is the furthest distance from any 2 points of the cloud
// 	// double diameter_model = computeCloudDiameter(model);
// 	// std::cout << "Diameter : " << diameter_model << std::endl;
// 	// //We set the params based on the diameter to have general purpose
// 	// samp_rad = t_sampling * diameter_model;
// 	// norm_rad = 2 * samp_rad;
// 	// Lvoxel = samp_rad;

// 	// //Voxel grid filter
// 	// pcl::VoxelGrid<pcl::PointXYZ> vg;
// 	// vg.setInputCloud(model);
// 	// vg.setLeafSize(samp_rad, samp_rad, samp_rad);
// 	// vg.setDownsampleAllData(false);
// 	// vg.filter(*model_keypoints);

// 	// // Calculate all the normals of the entire surface
// 	// pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
// 	// pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
// 	// pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
// 	// ne.setInputCloud(model_keypoints);
// 	// ne.setSearchSurface(model);
// 	// ne.setNumberOfThreads(8);
// 	// ne.setSearchMethod(tree);
// 	// ne.setRadiusSearch(norm_rad);
// 	// ne.compute(*normals);

// 	// pcl::concatenateFields(*model_keypoints, *normals, *model_keypoints_with_normals);

// 	// //Calculate PPF Descriptor of the model
// 	// pcl::PointCloud<pcl::PPFSignature>::Ptr descriptors_PPF = pcl::PointCloud<pcl::PPFSignature>::Ptr(new pcl::PointCloud<pcl::PPFSignature>());
// 	// pcl::PPFEstimation<pcl::PointNormal, pcl::PointNormal, pcl::PPFSignature> ppf_estimator;
// 	// ppf_estimator.setInputCloud(model_keypoints_with_normals);
// 	// ppf_estimator.setInputNormals(model_keypoints_with_normals);
// 	// ppf_estimator.compute(*descriptors_PPF);

// 	// ppf_hashmap_search->setInputFeatureCloud(descriptors_PPF);

// 	std::cout << "Done with Preparing Model Descriptor Offline....." << std::endl;
// 	// std::getchar();
// 	return true;
// }

// void DescriptorPPF::storeLatestCloud(const PointCloudType::ConstPtr &cloud)
// {
// 	latestCloud = cloud->makeShared();
// 	std::cout << "Cloud Update with Size " << latestCloud->points.size() << " ........." << std::endl;
// }
