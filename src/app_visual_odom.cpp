#include <string>

#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <thread>
#include <iomanip>

#include "optical_flow_vo/utils.hpp"
#include "optical_flow_vo/cam_loader.hpp"
#include "optical_flow_vo/optical_flow.hpp"
#include "Eigen/Core"

int main(int argc, char** argv) {
  const std::string about =
      "This sample demonstrates 2D Visual Odometry via optical flow from a "
      "downward facing camera.\n";
  const std::string keys =
      "{ h help |      | print this help message }"
      "{ @video | test.mp4 | path to video file }"
      "{ @config | config/params.yml | path to config file }";
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about(about);
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  CamLoader cam_loader(parser.get<std::string>("@video"), false, true);
  // std::thread cam_thread = cam_loader.StartCamThread();
  // std::cout << "Start display thread";
  // std::thread display_thread = cam_loader.StartDisplayThread();

  // Create Optical Flow Node
  OpticalFlowNode of_node(1280, 720, FeatureDetectorType::kORB);
  // Buffer 2 images
  std::vector<cv::Mat> buf;
  buf.reserve(2);
  buf.push_back(cam_loader.GetNextImage());
  double x_global = 0, y_global = 0, angle = 0;
  Eigen::Matrix3f T_global = Eigen::Matrix3f::Identity();
  while (true) {
    buf.push_back(cam_loader.GetNextImage());
    of_node.DetectKeypoints(buf[0]);

    of_node.CalcOpticalFlow(buf[0], buf[1]);

    // std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(10)
    // << of_node.OpticalFlow2Transform() << std::endl;
    Eigen::Matrix3f T = of_node.OpticalFlow2Transform();
    T_global = T * T_global;
    std::cout << "Current Location: X = " << T_global(0,2) << ", Y = " << T_global(1,2)
              << ", Angle = " << std::atan2(T_global(1,0),T_global(0,0)) << std::endl;
    cv::imshow("Display", buf[0]);
    cv::waitKey(1);
    buf.erase(buf.begin());
  }

  return 0;
}