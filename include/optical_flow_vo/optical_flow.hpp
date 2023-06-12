#ifndef DCC4E219_D8C4_4F5D_A3FB_A1A06F739560
#define DCC4E219_D8C4_4F5D_A3FB_A1A06F739560

#include "opencv2/features2d.hpp"
#include "optical_flow_vo/utils.hpp"

#include <memory>

class OpticalFlowNode{
    OpticalFlowNode(FeatureDetectorType type);
    ~OpticalFlowNode();

    private:
    std::unique_ptr<cv::FeatureDetector> feature_detector;
};

#endif /* DCC4E219_D8C4_4F5D_A3FB_A1A06F739560 */
