#ifndef PPF
#define PPF

#include "pclFunction.h"
#include "meshSampling.h"
#include "HPR.h"
#include "configReader.h"

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


struct PoseEstimationParams {
  // Preprocess
  float voxel_size;           // [m] kích thước voxel (ví dụ 0.005f)
  float normal_radius;        // [m] bán kính ước tính normal (ví dụ 0.01f)
  float feature_radius;       // [m] bán kính FPFH (ví dụ 0.025f)

  // RANSAC prerejective (coarse)
  int   sac_max_iters;        // ví dụ 50000
  int number_of_sample;
  int   sac_corr_randomness;  // k-láng giềng trong feature space, ví dụ 5
  float sac_sim_threshold;    // [0..1], ví dụ 0.9f
  float sac_max_corr_dist;    // [m], ví dụ 0.03f
  float sac_inlier_fraction;  // ví dụ 0.25f

  // ICP refine (point-to-plane)
  float icp_max_corr_dist;    // [m], ví dụ 0.02f
  int   icp_max_iters;        // ví dụ 50
  float icp_trans_eps;        // ví dụ 1e-6f
  float icp_fit_eps;          // ví dụ 1e-6f

  // Tuỳ chọn: đưa vào initial guess
  Eigen::Matrix4f T_init;
  bool use_external_init;

  PoseEstimationParams();
};

// Trả về ma trận biến đổi T sao cho scene ≈ T * object
// out_fitness: ICP fitness score (nhỏ hơn là tốt). Nếu RANSAC/ICP không hội tụ, trả về +inf.
// out_aligned: (tuỳ chọn) point cloud object đã áp T (độ phân giải cao).
Eigen::Matrix4f EstimatePoseRobustPCL(
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& object_in,
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& scene_in,
  const PoseEstimationParams& params,
  pcl::PointCloud<pcl::PointXYZ>::Ptr* out_coarse_aligned);

  // ----- Tham số multi-hypothesis -----
struct MultiHypothesisParams {
  int   num_hypotheses;           // số lần sinh coarse pose (ví dụ 32)
  int   refine_top_k;             // refine ICP cho K ứng viên tốt nhất (ví dụ 5)
  float cluster_trans_eps;        // ngưỡng gom cụm theo t (m), ví dụ 0.01
  float cluster_rot_deg;          // ngưỡng gom cụm theo góc (độ), ví dụ 5.0
  float eval_inlier_dist_coarse;  // ngưỡng inlier khi chấm coarse (m)
  float eval_inlier_dist_refine;  // ngưỡng inlier khi chấm refine (m)
  float source_subsample_ratio;   // tỉ lệ random subsample nguồn mỗi lần (0.5..1.0)
  unsigned int random_seed;       // seed cho RNG; có thể bỏ qua
  bool  use_fixed_seed;           // true -> dùng seed cố định cho tái lập

  MultiHypothesisParams();
};

// ----- Thông tin từng giả thuyết -----
struct PoseHypothesis {
  Eigen::Matrix4f T;          // transform ứng viên
  float coarse_inlier_ratio;  // tỉ lệ inlier sau coarse
  int   coarse_inliers;       // số inlier coarse
  float icp_fitness;          // fitness sau ICP (nếu có refine)
  int   refine_inliers;       // số inlier sau refine
  bool  refined;              // đã refine hay chưa

  PoseHypothesis()
  : T(Eigen::Matrix4f::Identity()),
    coarse_inlier_ratio(0.0f),
    coarse_inliers(0),
    icp_fitness(std::numeric_limits<float>::infinity()),
    refine_inliers(0),
    refined(false) {}
};

// ----- API: ước lượng pose đa giả thuyết -----
// Trả về T_best. Nếu muốn dừng ở coarse, đặt refine_top_k=0.
// out_all_hypotheses (tuỳ chọn) để xem mọi ứng viên & điểm số.
Eigen::Matrix4f EstimatePoseMultiHypothesisPCL(
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& object_in,
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& scene_in,
  const PoseEstimationParams& base_params,
  const MultiHypothesisParams& mh_params,
  float* out_best_fitness,
  pcl::PointCloud<pcl::PointXYZ>::Ptr* out_aligned,
  std::vector<PoseHypothesis>* out_all_hypotheses);


class DescriptorPPF {
public:
	DescriptorPPF();

	void setModelPath(std::string model_path);
	void setModelPcdPath(std::string model_path);

	bool loadModel();
	bool loadModelPCD();
	void saveToPCD(std::string path);

	void createSimScene(ConfigReader &cfg);

	void match(ConfigReader &cfg);

public:
	CustomVisualizer customViewer;

private:
	std::string model_dir;
	std::string model_pcd_dir;

	pcl::PointCloud<pcl::PointXYZ>::Ptr model_sampling = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
  
	pcl::PointCloud<pcl::PointXYZ>::Ptr model = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr scene = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr scene_ori = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());


	//Others
	pcl::console::TicToc tt;	// Tictoc for process-time calculation
	std::mutex mtx;
};

#endif
