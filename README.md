# Optical-Flow-Localisation

## Prerequisite

*   Eigen3
*   OpenCV
*   CMake

## Installation and build

    ```
    git clone https://github.com/oyqmatt/Optical-Flow-Localisation.git
    cd Optical-Flow-Localisation
    mkdir build 
    cd build
    cmake ..
    make
    ```

## How to use

### To save a output video file

Uncomment this line in optical_flow.cpp

    ```
    #define VID_OUT
    ```

In build directory

    ```
    ./Optical-Flow-Localisation <path-to-video>
    ```

## Issues to be fixed

* Poor tracking performanc in combined rotation and translation
* To implement video undistortion
