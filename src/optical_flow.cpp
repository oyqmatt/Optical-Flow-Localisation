#include "optical_flow_vo/optical_flow.hpp"

OpticalFlowNode::OpticalFlowNode(FeatureDetectorType type) {
switch (type)
{
case FeatureDetectorType::kBRIEF:
    feature_detector = cv::BR
    break;

default:
    break;
}
}