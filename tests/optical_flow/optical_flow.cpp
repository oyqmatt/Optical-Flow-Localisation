#include "optical_flow_vo/utils.hpp"
#include "optical_flow_vo/optical_flow.hpp"

int main() {
    FeatureDetectorType type = FeatureDetectorType::kBRISK;
    OpticalFlowNode of_node(type);
    return 0;
}