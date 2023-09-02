cmake_minimum_required(VERSION 3.0)
project(OpticalFlowVisualOdometry)

option(BUILD_TESTING "Build tests" OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# Dependencies
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
    src/app_visual_odom.cpp
    src/cam_loader.cpp
    src/optical_flow.cpp)
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${YAML_CPP_LIBRARIES}
    Threads::Threads
)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/config/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/config/)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
