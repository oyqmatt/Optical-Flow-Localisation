#ifndef OPTICAL_FLOW_VO_OPTICAL_FLOW
#define OPTICAL_FLOW_VO_OPTICAL_FLOW

#include <memory>

#include "Eigen/SparseCore"
#include "opencv2/features2d.hpp"
#include "opencv2/video/tracking.hpp"
#include "optical_flow_vo/utils.hpp"
#include "Eigen/SparseQR"

// Convenience for A matrix
typedef Eigen::Triplet<float> T;

class OpticalFlowNode {
 public:
  OpticalFlowNode(int width, int height, FeatureDetectorType type);
//   ~OpticalFlowNode();
  void DetectKeypoints(cv::Mat &input_img);
  void CalcOpticalFlow(cv::Mat &input_old, cv::Mat &input_new);
  Eigen::Matrix3f OpticalFlow2Transform();

 private:
  cv::Ptr<cv::FeatureDetector> feature_detector_;
  std::vector<cv::KeyPoint> kp0_;
  std::vector<cv::Point2f> p0_, p1_;
  std::vector<uint8_t> status_;
  int img_width_, img_height_;
  Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<int>>
      solver_;
};

#endif /* OPTICAL_FLOW_VO_OPTICAL_FLOW */
