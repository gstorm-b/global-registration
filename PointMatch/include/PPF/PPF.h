#ifndef PPF
#define PPF

#include "pclFunction.h"
#include "meshSampling.h"
#include "HPR.h"

#include <pcl/common/common.h>
#include <pcl/registration/boost_graph.h>
#include <pcl/registration/registration.h>
#include <pcl/features/ppf.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
// #include <numeric>
// #include <unordered_map>
#include <mutex>

class SymmetryPoseEstimator {
public:
  // Kiểu dữ liệu PCL rút gọn
  using PointT  = pcl::PointXYZ;
  using CloudT  = pcl::PointCloud<PointT>;
  using PointN  = pcl::PointNormal;
  using CloudN  = pcl::PointCloud<PointN>;
  struct FPFH33; // fwd (định nghĩa thực trong .cpp)

  struct Params {
    // I/O
    std::string model_path;
    std::string scene_path;
    std::string out_dir = "pose_out";

	pcl::PointCloud<pcl::PointXYZ>::Ptr model_pcd = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr scene_pcd = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());

    // Sampling & đặc trưng
    float voxel = 0.003f;                 // voxel cho cả model/scene
    std::optional<float> voxel_scene;     // nếu muốn khác voxel cho scene
    float normal_radius = 0.01f;
    float fpfh_radius = 0.025f;

    // Coarse (RANSAC FPFH)
    float ransac_dist = 0.01f;
    int   ransac_runs = 48;

    // Đối xứng quay Cn
    int   sym_n = 4;
    Eigen::Vector3f sym_axis = Eigen::Vector3f(0,0,1);
    float dedup_eps_deg = 3.0f;
    float dedup_tau = 0.003f;
    int   keep_topk = 10;

    // ICP
    float icp_corr = 0.01f;
    int   icp_iter = 60;

    // Scoring (visibility-aware)
    Eigen::Vector3f cam = Eigen::Vector3f(0,0,0);
    float trunc = 0.01f;
    bool  export_all = false;

    // STL sampling (nếu đọc STL)
    float stl_sample_radius = 0.001f; // bán kính cho UniformSampling trên vertices
  };

  struct Candidate {
    int   idx = -1;
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    double icp_fitness = 0;
    double icp_rmse = 0;
    double vis_rmse = 0;  // điểm chính để chọn (nhỏ là tốt)
  };

  struct Result {
    bool success = false;
    Eigen::Matrix4f T_best = Eigen::Matrix4f::Identity();
    std::vector<Candidate> all;   // các ứng viên sau refine + scoring
    std::string out_dir;          // nơi đã lưu file
  };

  struct Hypo { Eigen::Matrix4f T; float score; };

public:
  explicit SymmetryPoseEstimator(const Params& p);
  ~SymmetryPoseEstimator() = default;

  // Chạy toàn bộ pipeline và trả về kết quả
  Result run();

private:
  // ===== Helper nội bộ =====
  // I/O
  bool loadCloudAny(const std::string& path, CloudT::Ptr& out) const;
  CloudT::Ptr voxelDown(const CloudT::Ptr& in, float voxel) const;

  // Normals & FPFH
  CloudN::Ptr toWithNormals(const CloudT::Ptr& in, float normal_radius, const Eigen::Vector3f& view) const;
  std::shared_ptr<pcl::PointCloud<FPFH33>> computeFPFH(const CloudN::Ptr& in, float radius) const;

  // Coarse hypotheses
  
  std::vector<Hypo> generateMultiHypotheses(const CloudN::Ptr& model_n,
                                            const CloudN::Ptr& scene_n,
                                            const std::shared_ptr<pcl::PointCloud<FPFH33>>& f_model,
                                            const std::shared_ptr<pcl::PointCloud<FPFH33>>& f_scene,
                                            float base_dist, int runs) const;

  // Đối xứng & dedup
  static float so3Geodesic(const Eigen::Matrix3f& Ra, const Eigen::Matrix3f& Rb);
  static std::vector<Eigen::Matrix3f> makeCn(const Eigen::Vector3f& axis, int n);
  static bool equivalentUnderSym(const Eigen::Matrix3f& Ra, const Eigen::Matrix3f& Rb,
                                 const std::vector<Eigen::Matrix3f>& G, float eps_deg);
  static std::vector<Hypo> dedupHypotheses(const std::vector<Hypo>& hyps,
                                           const std::vector<Eigen::Matrix3f>& G,
                                           float eps_deg, float tau, int topk);

  // ICP
  Eigen::Matrix4f icpRefinePointToPlane(const CloudN::Ptr& model_n,
                                        const CloudN::Ptr& scene_n,
                                        const Eigen::Matrix4f& T0,
                                        double& fitness_out, double& rmse_out) const;

  // Scoring
  CloudT::Ptr hiddenVisibleSubset(const CloudT::Ptr& pcd, const Eigen::Vector3f& cam,
                                  float radius_scale=1.2f) const;
  double visibilityAwareScore(const CloudT::Ptr& model_xyz,
                              const CloudT::Ptr& scene_xyz,
                              const Eigen::Matrix4f& T,
                              const Eigen::Vector3f& cam,
                              double trunc,
                              CloudT::Ptr* visible_out=nullptr) const;

  // Save
  static void saveTransformTxt(const std::string& path, const Eigen::Matrix4f& T);
  static void ensureDir(const std::string& path);

private:
  Params params_;
};

class DescriptorPPF {
public:
	DescriptorPPF();

	void setModelPath(std::string model_path);
	void setModelPcdPath(std::string model_path);

	bool loadModel();
	bool loadModelPCD();
	void saveToPCD(std::string path);

	void createSimScene();

	void match();

public:
	CustomVisualizer customViewer;

	
	
private:
	std::string model_dir;
	std::string model_pcd_dir;

	pcl::PointCloud<pcl::PointXYZ>::Ptr model = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_sampling = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());

	pcl::PointCloud<pcl::PointXYZ>::Ptr scene = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());


	//Others
	pcl::console::TicToc tt;	// Tictoc for process-time calculation
	std::mutex mtx;
};

#endif
