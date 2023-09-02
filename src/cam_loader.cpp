#include "optical_flow_vo/cam_loader.hpp"
#include <iostream>
#include "yaml-cpp/yaml.h"
#include <iomanip>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

CamLoader::CamLoader(std::string config_file, bool undistort, bool video_out) {
  // Load yaml file
  YAML::Node config = YAML::LoadFile(config_file);
  const std::string img_src =
      config["camera"]["video_source"].as<std::string>();

  // Load img size
  if (config["camera"]["image_size"].IsDefined()) {
    img_src_.set(cv::CAP_PROP_FRAME_WIDTH, config["camera"]["width"].as<int>());
    img_src_.set(cv::CAP_PROP_FRAME_HEIGHT,
                 config["camera"]["width"].as<int>());
  }

//   img_src_.open(img_src, cv::CAP_V4L);
  img_src_.open(img_src, cv::CAP_FFMPEG);
  if (!img_src_.isOpened()) {
    std::cout << "Could not open video file: " << img_src << std::endl;
    exit(1);
  }

  // Load camera matrix
  if (config["camera"]["camera_matrix"].IsDefined()) {
    cam_mtx_ = cv::Mat::eye(3, 3, CV_32FC1);
    cam_mtx_.at<float>(0, 0) = 3157;
    cam_mtx_.at<float>(0, 1) = 5.4478;
    cam_mtx_.at<float>(0, 2) = 1858;
    cam_mtx_.at<float>(1, 0) = 0;
    cam_mtx_.at<float>(1, 1) = 3156;
    cam_mtx_.at<float>(1, 2) = 1200;
    std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(4)
              << "Camera Matrix:\n"
              << cam_mtx_ << std::endl;

    // Load distortion coefficients only if camera matrix is defined
    if (config["camera"]["distortion_coefficients"].IsDefined()) {
      dist_coeffs_ =
          config["camera"]["distortion_coefficients"].as<std::vector<double>>();
      std::cout << "Distortion Coefficients:\n";
      for (auto &coeff : dist_coeffs_) {
        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(4)
                  << "\t" << coeff;
      }
      std::cout << std::endl;
    }
  }
}

std::thread CamLoader::StartDisplayThread() {
  return std::thread(&CamLoader::DisplayThread, this);
  //   display_thread.join();
}

void CamLoader::DisplayThread() {
  cv::namedWindow("Display", cv::WINDOW_AUTOSIZE);
  while (true) {
    cv::imshow("Display", img_);
    cv::waitKey(10);
  }
}

/**
 * @brief Start thread for real time camera feed
 *
 */
std::thread CamLoader::StartCamThread() {
  GetNextImage();
  return std::thread(&CamLoader::GrabImage, this);
}

void CamLoader::GrabImage() {
  while (true) {
    img_src_.grab();
  }
}
cv::Mat CamLoader::GetLatestImage() {
  cv::Mat temp;
  img_src_.retrieve(temp);
  cv::cvtColor(temp, img_, cv::COLOR_BGR2GRAY);
  return img_.clone();
}

cv::Mat CamLoader::GetNextImage() {
  img_src_.read(img_);
  return img_.clone();
}
