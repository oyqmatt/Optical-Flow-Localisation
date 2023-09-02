#ifndef OPTICAL_FLOW_VO_CAM_LOADER
#define OPTICAL_FLOW_VO_CAM_LOADER

#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc.hpp"
#include <vector>
#include <string>
#include <iomanip>
#include <thread>

class CamLoader
{
public:
    CamLoader(std::string config_file, bool undistort, bool video_out = true);
    // ~CamLoader();
    // Realtime
    std::thread StartCamThread();
    cv::Mat GetLatestImage();
    // Non-realtime
    cv::Mat GetNextImage();
    std::thread StartDisplayThread();

private:
    // For realtime
    void GrabImage();
    void DisplayThread();
    cv::VideoCapture img_src_;
    std::vector<double> dist_coeffs_;
    cv::Mat cam_mtx_;
    cv::Mat img_;
};

#endif /* OPTICAL_FLOW_VO_CAM_LOADER */
