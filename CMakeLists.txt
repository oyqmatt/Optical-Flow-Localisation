cmake_minimum_required(VERSION 3)
project( OpticalFlowLocalisation )

# Dependencies
find_package( OpenCV REQUIRED )
find_package(Eigen3 REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable( OpticalFlowLocalisation src/main.cpp )
target_link_libraries( OpticalFlowLocalisation ${OpenCV_LIBS} )
