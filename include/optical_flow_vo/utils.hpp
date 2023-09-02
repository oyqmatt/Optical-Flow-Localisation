#ifndef OPTICAL_FLOW_VO_UTILS_HPP
#define OPTICAL_FLOW_VO_UTILS_HPP

// ORB and FAST have same keypoint detector
enum class FeatureDetectorType {
  kBRISK,
  kORB,
  // kMSER,
  kFAST = kORB,
  kAGAST,
  kGFTT,
  kBLOB,
  // kKAZE,
  // kAKAZE,
};

#endif /* OPTICAL_FLOW_VO_UTILS_HPP */
