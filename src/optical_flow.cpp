#include "optical_flow_vo/optical_flow.hpp"

#include <iostream>

#include "Eigen/SparseQR"

OpticalFlowNode::OpticalFlowNode(int width, int height,
                                 FeatureDetectorType type)
    : img_width_(width), img_height_(height) {
  switch (type) {
    case FeatureDetectorType::kORB:
      feature_detector_ = cv::ORB::create();
      break;
    case FeatureDetectorType::kBRISK:
      feature_detector_ = cv::BRISK::create();
      break;
    case FeatureDetectorType::kAGAST:
      feature_detector_ = cv::AgastFeatureDetector::create();
      break;
    case FeatureDetectorType::kGFTT:
      feature_detector_ = cv::GFTTDetector::create();
      break;
    case FeatureDetectorType::kBLOB:
      feature_detector_ = cv::SimpleBlobDetector::create();
      break;
    default:
      std::cout << "Invalid feature detector type, using ORB" << std::endl;
      feature_detector_ = cv::ORB::create();
      break;
  }
}

/**
 * @brief Detect keypoints and convert to Points
 *
 * @param input_img
 */
void OpticalFlowNode::DetectKeypoints(cv::Mat &input_img) {
  feature_detector_->detect(input_img, kp0_);
  cv::KeyPoint::convert(kp0_, p0_);
  // std::cout << "p0: " << p0_.size() << " keypoints" << std::endl;
}

/**
 * @brief Calculate optical flow
 *
 * @param input_old
 * @param input_new
 */
void OpticalFlowNode::CalcOpticalFlow(cv::Mat &input_old, cv::Mat &input_new) {
  std::vector<uint8_t> status;
  std::vector<float> err;
  cv::TermCriteria criteria = cv::TermCriteria(
      (cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
  cv::calcOpticalFlowPyrLK(input_old, input_new, p0_, p1_, status_, err,
                           cv::Size(15, 15), 2, criteria);
  // std::cout << "p1: " << p1_.size() << " keypoints" << std::endl;
}

/**
 * @brief Returns 3D Transformation (SO3) matrix
 *
 * @return Eigen::Matrix3f
 */
Eigen::Matrix3f OpticalFlowNode::OpticalFlow2Transform() {
  // Initialise Ax=b variables, use tripletList to create sparse matrix later
  Eigen::VectorXf b(2 * p0_.size());
  Eigen::VectorXf x(3);
  std::vector<T> tripletList;
  uint i, j = 0;
  for (i = 0; i < p0_.size(); i++) {
    // Select good points
    if (status_[i] == 1) {
      // Orthogonal Components. Assume rotation about center
      double disp_x = p0_[i].x - img_width_ / 2;
      double disp_y = img_height_ - p0_[i].y - img_height_ / 2;
      // Coordinate transform for camera to 2d base frame
      double x_orth = -disp_y;
      double y_orth = -disp_x;

      // Append A and B matrix
      b[j] = p1_[i].x - p0_[i].x;
      b[j + 1] = p0_[i].y - p1_[i].y;

      tripletList.push_back(T(j, 0, 1));
      tripletList.push_back(T(j, 2, x_orth));
      tripletList.push_back(T(j + 1, 1, 1));
      tripletList.push_back(T(j + 1, 2, y_orth));
      j += 2;
    }
  }
  // std::cout << "Detected " << j/2 << " good points" << std::endl;
  b.conservativeResize(j);
  Eigen::SparseMatrix<float> A(j, 3);
  A.setFromTriplets(tripletList.begin(), tripletList.end());

  // Invert Matrix here
  solver_.compute(A);
  if (solver_.info() != Eigen::Success) {
    // decomposition failed
    std::cerr << "Decomposition failed" << std::endl;
  }
  x = solver_.solve(b);
  if (solver_.info() != Eigen::Success) {
    // solving failed
    std::cerr << "Solving failed!" << std::endl;
  }
  Eigen::Matrix3f T;
  T << cos(x[2]), -sin(x[2]), x[0], sin(x[2]), cos(x[2]), x[1], 0, 0, 1;
  return T;
}
