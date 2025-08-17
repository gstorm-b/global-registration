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

#ifdef _WIN32
  #include <direct.h>
#endif

// Định nghĩa kiểu FPFH33
struct SymmetryPoseEstimator::FPFH33 : public pcl::FPFHSignature33 {};

SymmetryPoseEstimator::SymmetryPoseEstimator(const Params& p) : params_(p) {}

// ====== Tiện ích ======
void SymmetryPoseEstimator::ensureDir(const std::string& p) {
#ifdef _WIN32
  _mkdir(p.c_str());
#else
  ::mkdir(p.c_str(), 0755);
#endif
}

void SymmetryPoseEstimator::saveTransformTxt(const std::string& path, const Eigen::Matrix4f& T) {
  std::ofstream ofs(path);
  ofs << std::fixed;
  for (int r=0;r<4;++r) {
    for (int c=0;c<4;++c) {
      ofs << T(r,c);
      if (c<3) ofs << " ";
    }
    ofs << "\n";
  }
}

static bool ends_with(const std::string& s, const std::string& suf) {
  if (s.size() < suf.size()) return false;
  return std::equal(suf.rbegin(), suf.rend(), s.rbegin(),
                    [](char a, char b){ return std::tolower(a)==std::tolower(b); });
}

#ifdef PCL_IO_STL
  #include <pcl/io/vtk_lib_io.h>
  #include <pcl/filters/uniform_sampling.h>
  static bool mesh_to_points_uniform(const pcl::PolygonMesh& mesh,
                                     SymmetryPoseEstimator::CloudT::Ptr& out,
                                     float radius)
  {
    SymmetryPoseEstimator::CloudT::Ptr verts(new SymmetryPoseEstimator::CloudT);
    pcl::fromPCLPointCloud2(mesh.cloud, *verts);
    if (verts->empty()) return false;
    pcl::UniformSampling<SymmetryPoseEstimator::PointT> us;
    us.setInputCloud(verts);
    us.setRadiusSearch(radius);
    pcl::PointCloud<int> sampled_idx;
    us.compute(sampled_idx);
    out.reset(new SymmetryPoseEstimator::CloudT);
    out->reserve(sampled_idx.size());
    for (int id : sampled_idx.points) out->push_back((*verts)[id]);
    return !out->empty();
  }
#endif

bool SymmetryPoseEstimator::loadCloudAny(const std::string& path, CloudT::Ptr& out) const {
  out.reset(new CloudT);
  if (ends_with(path, ".pcd")) {
    return pcl::io::loadPCDFile<PointT>(path, *out) == 0;
  } else if (ends_with(path, ".ply")) {
    return pcl::io::loadPLYFile<PointT>(path, *out) == 0;
  } else if (ends_with(path, ".xyz")) {
    std::ifstream ifs(path);
    if (!ifs.good()) return false;
    PointT p; while (ifs >> p.x >> p.y >> p.z) out->push_back(p);
    return !out->empty();
  } else if (ends_with(path, ".stl")) {
#ifdef PCL_IO_STL
    pcl::PolygonMesh mesh;
    if (pcl::io::loadPolygonFileSTL(path, mesh)==0) {
      pcl::console::print_error("STL load failed: %s\n", path.c_str());
      return false;
    }
    return mesh_to_points_uniform(mesh, out, params_.stl_sample_radius);
#else
    pcl::console::print_error("PCL không build STL/VTK, không thể đọc STL.\n");
    return false;
#endif
  }
  pcl::console::print_error("Định dạng không hỗ trợ: %s\n", path.c_str());
  return false;
}

SymmetryPoseEstimator::CloudT::Ptr
SymmetryPoseEstimator::voxelDown(const CloudT::Ptr& in, float voxel) const {
  if (voxel <= 0) return in;
  pcl::VoxelGrid<PointT> vg;
  vg.setInputCloud(in);
  vg.setLeafSize(voxel, voxel, voxel);
  CloudT::Ptr out(new CloudT);
  vg.filter(*out);
  return out;
}

SymmetryPoseEstimator::CloudN::Ptr
SymmetryPoseEstimator::toWithNormals(const CloudT::Ptr& in, float normal_radius, const Eigen::Vector3f& view) const {
  pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
  ne.setInputCloud(in);
  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(normal_radius);
  ne.setViewPoint(view.x(), view.y(), view.z());
  pcl::PointCloud<pcl::Normal>::Ptr ns(new pcl::PointCloud<pcl::Normal>);
  ne.compute(*ns);

  CloudN::Ptr out(new CloudN);
  out->resize(in->size());
  for (size_t i=0;i<in->size();++i) {
    PointN pn;
    pn.x = (*in)[i].x; pn.y = (*in)[i].y; pn.z = (*in)[i].z;
    pn.normal_x = (*ns)[i].normal_x;
    pn.normal_y = (*ns)[i].normal_y;
    pn.normal_z = (*ns)[i].normal_z;
    (*out)[i] = pn;
  }
  return out;
}

std::shared_ptr<pcl::PointCloud<SymmetryPoseEstimator::FPFH33>>
SymmetryPoseEstimator::computeFPFH(const CloudN::Ptr& in, float radius) const {
  pcl::FPFHEstimationOMP<PointN, PointN, FPFH33> est;
  est.setInputCloud(in);
  est.setInputNormals(in);
  pcl::search::KdTree<PointN>::Ptr kdt(new pcl::search::KdTree<PointN>);
  est.setSearchMethod(kdt);
  est.setRadiusSearch(radius);
  auto out = std::make_shared<pcl::PointCloud<FPFH33>>();
  est.compute(*out);
  return out;
}

// ====== Toán học đối xứng ======
float SymmetryPoseEstimator::so3Geodesic(const Eigen::Matrix3f& Ra, const Eigen::Matrix3f& Rb) {
  Eigen::Matrix3f R = Ra * Rb.transpose();
  float c = 0.5f * (R.trace() - 1.0f);
  c = std::max(-1.0f, std::min(1.0f, c));
  return std::acos(c);
}

std::vector<Eigen::Matrix3f> SymmetryPoseEstimator::makeCn(const Eigen::Vector3f& axis, int n) {
  std::vector<Eigen::Matrix3f> G; G.reserve(std::max(1,n));
  Eigen::Vector3f ax = axis.normalized();
  for (int k=0;k<std::max(1,n);++k) {
    float th = 2.0f * float(M_PI) * float(k) / float(std::max(1,n));
    Eigen::AngleAxisf aa(th, ax);
    G.push_back(aa.toRotationMatrix());
  }
  return G;
}

bool SymmetryPoseEstimator::equivalentUnderSym(const Eigen::Matrix3f& Ra, const Eigen::Matrix3f& Rb,
                                               const std::vector<Eigen::Matrix3f>& G, float eps_deg) {
  float eps = eps_deg * float(M_PI) / 180.0f;
  for (const auto& S : G) {
    float d = so3Geodesic(Ra, Rb * S);
    if (d < eps) return true;
  }
  return false;
}

std::vector<SymmetryPoseEstimator::Hypo>
SymmetryPoseEstimator::dedupHypotheses(const std::vector<Hypo>& hyps,
                                       const std::vector<Eigen::Matrix3f>& G,
                                       float eps_deg, float tau, int topk)
{
  std::vector<Hypo> sorted = hyps;
  std::sort(sorted.begin(), sorted.end(),
            [](const Hypo& a, const Hypo& b){ return a.score > b.score; });

  std::vector<Hypo> kept;
  for (const auto& h : sorted) {
    Eigen::Matrix3f R = h.T.block<3,3>(0,0);
    Eigen::Vector3f t = h.T.block<3,1>(0,3);
    bool ok = true;
    for (const auto& g : kept) {
      Eigen::Matrix3f Rg = g.T.block<3,3>(0,0);
      Eigen::Vector3f tg = g.T.block<3,1>(0,3);
      if ( (t - tg).norm() < tau && equivalentUnderSym(R, Rg, G, eps_deg) ) {
        ok = false; break;
      }
    }
    if (ok) {
      kept.push_back(h);
      if (topk > 0 && (int)kept.size() >= topk) break;
    }
  }
  return kept;
}

// ====== Coarse (Prerejective) ======
static SymmetryPoseEstimator::Hypo prerejective_once(
    const SymmetryPoseEstimator::CloudN::Ptr& model_n,
    const SymmetryPoseEstimator::CloudN::Ptr& scene_n,
    const std::shared_ptr<pcl::PointCloud<SymmetryPoseEstimator::FPFH33>>& f_model,
    const std::shared_ptr<pcl::PointCloud<SymmetryPoseEstimator::FPFH33>>& f_scene,
    float max_corr)
{
  pcl::SampleConsensusPrerejective<SymmetryPoseEstimator::PointN,
                                   SymmetryPoseEstimator::PointN,
                                   SymmetryPoseEstimator::FPFH33> sac;
  sac.setInputSource(model_n);
  sac.setSourceFeatures(f_model);
  sac.setInputTarget(scene_n);
  sac.setTargetFeatures(f_scene);
  sac.setMaximumIterations(20000);
  sac.setNumberOfSamples(4);
  sac.setCorrespondenceRandomness(60);
  sac.setSimilarityThreshold(0.9f);
  sac.setMaxCorrespondenceDistance(max_corr);
  sac.setInlierFraction(0.25f);

  SymmetryPoseEstimator::CloudN::Ptr aligned(new SymmetryPoseEstimator::CloudN);
  sac.align(*aligned);

  SymmetryPoseEstimator::Hypo h{};
  h.T = Eigen::Matrix4f::Identity();
  h.score = 0.0f;
  if (sac.hasConverged()) {
    h.T = sac.getFinalTransformation();
    // Heuristic tạm: inliers/fitness
    float denom = std::max(1e-6f, (float)sac.getFitnessScore());
    h.score = float(sac.getInliers().size()) / denom;
  }
  return h;
}

std::vector<SymmetryPoseEstimator::Hypo>
SymmetryPoseEstimator::generateMultiHypotheses(const CloudN::Ptr& model_n,
                                               const CloudN::Ptr& scene_n,
                                               const std::shared_ptr<pcl::PointCloud<FPFH33>>& f_model,
                                               const std::shared_ptr<pcl::PointCloud<FPFH33>>& f_scene,
                                               float base_dist, int runs) const
{
  std::mt19937 rng(0u);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f);
  std::vector<Hypo> hyps; hyps.reserve(runs);
  for (int i=0;i<runs;++i) {
    float thr = std::max(1e-6f, base_dist * (1.0f + 0.4f * U(rng)));
    hyps.emplace_back(prerejective_once(model_n, scene_n, f_model, f_scene, thr));
  }
  return hyps;
}

// ====== ICP ======
Eigen::Matrix4f SymmetryPoseEstimator::icpRefinePointToPlane(
    const CloudN::Ptr& model_n, const CloudN::Ptr& scene_n,
    const Eigen::Matrix4f& T0, double& fitness_out, double& rmse_out) const
{
  pcl::IterativeClosestPoint<PointN, PointN> icp;
  auto est = std::make_shared<pcl::registration::TransformationEstimationPointToPlaneLLS<PointN, PointN>>();
//   icp.setTransformationEstimation(est);
  icp.setInputSource(model_n);
  icp.setInputTarget(scene_n);
  icp.setMaxCorrespondenceDistance(params_.icp_corr);
  icp.setMaximumIterations(params_.icp_iter);
  icp.setUseReciprocalCorrespondences(false);

  CloudN::Ptr src(new CloudN(*model_n)), aligned(new CloudN);
  pcl::transformPointCloudWithNormals(*src, *src, T0);
  icp.align(*aligned);

  Eigen::Matrix4f T = T0;
  if (icp.hasConverged()) {
    T = icp.getFinalTransformation() * T0;
    fitness_out = icp.getFitnessScore();
    rmse_out = icp.getFitnessScore();
  } else {
    fitness_out = 1e9; rmse_out = 1e9;
  }
  return T;
}

// ====== Visibility-aware scoring ======
SymmetryPoseEstimator::CloudT::Ptr
SymmetryPoseEstimator::hiddenVisibleSubset(const CloudT::Ptr& pcd, const Eigen::Vector3f& cam, float radius_scale) const {
  if (pcd->empty()) return pcd;
  Eigen::Vector4f min_pt, max_pt;
  pcl::getMinMax3D(*pcd, min_pt, max_pt);
  float rad = 0.5f * (max_pt.head<3>() - min_pt.head<3>()).norm();
  float radius = rad * radius_scale;

  std::vector<int> vis_idx;
//   pcl::HiddenPointRemoval<PointT> hpr;
//   hpr.setInputCloud(pcd);
//   hpr.setViewPoint(cam.x(), cam.y(), cam.z());
//   hpr.setRadius(radius);
//   hpr.compute(vis_idx);

  CloudT::Ptr out(new CloudT);
  out->reserve(vis_idx.size());
  for (int id : vis_idx) out->push_back((*pcd)[id]);
  return out;
}

static double chamfer_one_sided_trunc(const SymmetryPoseEstimator::CloudT::Ptr& src,
                                      pcl::KdTreeFLANN<SymmetryPoseEstimator::PointT>& kd,
                                      double trunc)
{
  if (src->empty()) return 1e6;
  double acc = 0.0;
  std::vector<int> idx(1);
  std::vector<float> dist2(1);
  for (const auto& p : src->points) {
    if (kd.nearestKSearch(p, 1, idx, dist2) > 0) {
      double d = std::sqrt(dist2[0]);
      if (d > trunc) d = trunc;
      acc += d;
    } else acc += trunc;
  }
  return acc / std::max<size_t>(1, src->size());
}

double SymmetryPoseEstimator::visibilityAwareScore(
    const CloudT::Ptr& model_xyz, const CloudT::Ptr& scene_xyz,
    const Eigen::Matrix4f& T, const Eigen::Vector3f& cam, double trunc, CloudT::Ptr* visible_out) const
{
  CloudT::Ptr model_tf(new CloudT);
  pcl::transformPointCloud(*model_xyz, *model_tf, T);

  CloudT::Ptr vis = hiddenVisibleSubset(model_tf, cam);
  if (visible_out) *visible_out = vis;

  pcl::KdTreeFLANN<PointT> kd; kd.setInputCloud(scene_xyz);
  return chamfer_one_sided_trunc(vis, kd, trunc);
}

// ====== RUN PIPELINE ======
SymmetryPoseEstimator::Result SymmetryPoseEstimator::run() {
	Result R; R.out_dir = params_.out_dir;
	ensureDir(params_.out_dir);

	// Load
	CloudT::Ptr model_raw(new CloudT), scene_raw(new CloudT);
	if (!loadCloudAny(params_.model_path, model_raw) || !loadCloudAny(params_.scene_path, scene_raw)) {
		pcl::console::print_error("Load model/scene thất bại.\n");
		return R;
	}

	const float voxel_scene = params_.voxel_scene ? *params_.voxel_scene : params_.voxel;
	auto model = voxelDown(model_raw, params_.voxel);
	auto scene = voxelDown(scene_raw, voxel_scene);

	// Normals + FPFH
	auto model_n = toWithNormals(model, params_.normal_radius, params_.cam);
	auto scene_n = toWithNormals(scene, params_.normal_radius, params_.cam);
	auto f_model = computeFPFH(model_n, params_.fpfh_radius);
	auto f_scene = computeFPFH(scene_n, params_.fpfh_radius);

	// Coarse
	auto hyps_all = generateMultiHypotheses(model_n, scene_n, f_model, f_scene,
											params_.ransac_dist, params_.ransac_runs);

	// Dedup theo đối xứng
	auto G = makeCn(params_.sym_axis, std::max(1, params_.sym_n));
	auto hyps = dedupHypotheses(hyps_all, G, params_.dedup_eps_deg, params_.dedup_tau, params_.keep_topk);

	if (hyps.empty()) {
	pcl::console::print_warn("Không có giả thuyết coarse hợp lệ sau dedup.\n");
		return R;	
	}

	// ICP + scoring
	std::vector<Candidate> refined; refined.reserve(hyps.size());

	// Chuẩn bị cloud XYZ cho scoring
	CloudT::Ptr model_xyz(new CloudT); pcl::copyPointCloud(*model_n, *model_xyz);
	CloudT::Ptr scene_xyz = scene;

	double best_score = 1e18; int best_id = -1;

	for (size_t i=0;i<hyps.size();++i) {
	const auto& h = hyps[i];
	double fit=0, rmse=0;
	Eigen::Matrix4f Ticp = icpRefinePointToPlane(model_n, scene_n, h.T, fit, rmse);

	SymmetryPoseEstimator::CloudT::Ptr vis;
	double vis_rmse = visibilityAwareScore(model_xyz, scene_xyz, Ticp, params_.cam, params_.trunc,
											params_.export_all ? &vis : nullptr);

	Candidate c; c.idx = int(i); c.T = Ticp; c.icp_fitness = fit; c.icp_rmse = rmse; c.vis_rmse = vis_rmse;
	refined.push_back(c);

	if (params_.export_all) {
		CloudT::Ptr model_aligned(new CloudT);
		pcl::transformPointCloud(*model_xyz, *model_aligned, Ticp);
		pcl::io::savePLYFileBinary(params_.out_dir + "/model_aligned_" + std::to_string(i) + ".ply", *model_aligned);
		pcl::io::savePLYFileBinary(params_.out_dir + "/model_visible_" + std::to_string(i) + ".ply", *vis);
	}
	if (vis_rmse < best_score) { best_score = vis_rmse; best_id = int(i); }
	}

	if (best_id < 0) {
	pcl::console::print_error("Không tìm được pose đáng tin.\n");
	return R;
	}

	// Lưu best
	auto best = refined[best_id];
	CloudT::Ptr model_best(new CloudT);
	pcl::transformPointCloud(*model, *model_best, best.T);
	pcl::io::savePLYFileBinary(params_.out_dir + "/model_aligned_best.ply", *model_best);
	saveTransformTxt(params_.out_dir + "/T_best.txt", best.T);

	R.success = true;
	R.T_best = best.T;
	R.all = std::move(refined);
	return R;
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

void DescriptorPPF::createSimScene() {
    // Move by 10 mm in X, rotate 30 deg around Z, pivot at (0,0,0)
    float tx = 0.010f, ty = 0.0f, tz = 0.05f;
    float roll = 0.0f, pitch = 0.0f, yaw = static_cast<float>(M_PI/6.0); // 30 deg
    Eigen::Vector3f pivot(0.f, 0.f, 0.f);

	pcl::PointCloud<pcl::PointXYZ>::Ptr dst(new pcl::PointCloud<pcl::PointXYZ>());
    transformCloudRPY(*model, *dst, tx, ty, tz, roll, pitch, yaw, pivot);

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_HPR(new pcl::PointCloud<pcl::PointXYZ>());
	HPR(dst, {0, 0, 0}, 3, cloud_xyz_HPR); //calculate from i view with 

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
	customViewer.viewer->addPointCloud(model);
	std::getchar();
}

void DescriptorPPF::match() {
	SymmetryPoseEstimator::Params p;
	p.model_path = model_pcd_dir;
	p.scene_path = model_pcd_dir;

	// Tuỳ biến nhanh (bạn chỉnh theo thực tế)
	p.out_dir = "pose_out";
	p.voxel = 0.003f;
	p.normal_radius = 0.01f;
	p.fpfh_radius = 0.025f;
	p.ransac_dist = 0.01f;
	p.ransac_runs = 64;

	p.sym_n = 6;                 // bậc đối xứng Cn
	p.sym_axis = {0,0,1};        // trục đối xứng
	p.dedup_eps_deg = 3.0f;
	p.dedup_tau = 0.003f;
	p.keep_topk = 12;

	p.icp_corr = 0.01f;
	p.icp_iter = 80;

	p.cam = {0,0,0};             // vị trí camera trong hệ scene
	p.trunc = 0.01f;
	p.export_all = true;

	SymmetryPoseEstimator solver(p);
	auto res = solver.run();

	if (!res.success) {
		std::cerr << "Pose estimation thất bại.\n";
		return;
	}

	std::cout << "Best T (model -> scene):\n" << res.T_best << "\n";
	std::cout << "Đã lưu vào thư mục: " << res.out_dir << "\n";
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
