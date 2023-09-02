add_executable(test_optical_flow optical_flow/optical_flow.cpp)

target_link_libraries(test_optical_flow
    ${OpenCV_LIBS})

target_include_directories(test_optical_flow
    ${OpenCV_INCLUDE_DIRS}
)