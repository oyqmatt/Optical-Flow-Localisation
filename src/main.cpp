#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/opencv.hpp"
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>

#define VID_OUT

using namespace cv;
using namespace std;
int main(int argc, char **argv)
{
    const string about =
        "This sample demonstrates Lucas-Kanade Optical Flow calculation.\n"
        "The example file can be downloaded from:\n"
        "  https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4";
    const string keys =
        "{ h help |      | print this help message }"
        "{ @image | vtest.avi | path to image file }";
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    string filename = samples::findFile(parser.get<string>("@image"));
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    VideoCapture capture(filename);
    if (!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open file!" << endl;
        return 0;
    }
    // Create some random colors
    vector<Scalar> colors;
    RNG rng;
    for(int i = 0; i < 900; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r,g,b));
    }
    Mat distorted, old_frame, old_gray, frame, frame_gray,R;
    vector<KeyPoint> kp0;
    vector<Point2f> p0, p1;

    // Camera Parameters
    //Mat cam_mtx = (Mat_<float>(3,3) << 8787.6 / 5472 * 1280, -1.2473, 2794.8 / 5472 * 1280,
    //                      0, 8769.4 / 3648  * 720, 1873.2 / 3648  * 720,
    //                      0, 0, 1);

    cv::Mat cam_mtx;
    cam_mtx = cv::Mat::eye(3,3,CV_32FC1);
    cam_mtx.at<float>(0,0) = 3157;
    cam_mtx.at<float>(0,1) = 5.4478;
    cam_mtx.at<float>(0,2) = 1858;
    cam_mtx.at<float>(1,0) = 0;
    cam_mtx.at<float>(1,1) = 3156;
    cam_mtx.at<float>(1,2) = 1200;

    
    //Mat dist_coeff = (Mat_<float>(5,1) << 0.5566, -16.7126, -0.0042, 0.0046, 221.9779);

    cv::Mat distCoeff;
    distCoeff = cv::Mat::zeros(5,1,CV_64FC1);

    // indices: k1, k2, p1, p2, k3, k4, k5, k6 
    // your coefficients here!
    double k1 = -0.0159;
    double k2 = -1.1865;
    double p11 = -0.0098;
    double p2 = 0.0064;
    double k3 = 3.3476;

    distCoeff.at<double>(0,0) = k1;
    distCoeff.at<double>(1,0) = k2;
    distCoeff.at<double>(2,0) = p11;
    distCoeff.at<double>(3,0) = p2;
    distCoeff.at<double>(4,0) = k3;

    Mat new_cam_mtx = (Mat_<float>(3,3) << 0,0,0,0,0,0,0,0,0);

    Mat map1, map2;

    // Initial frame
    capture >> old_frame;
    
    //cv::initUndistortRectifyMap(cam_mtx,distCoeff,R, new_cam_mtx,distorted.size(),CV_32FC1,map1,map2);
    //cv::remap(distorted,old_frame, map1,map2,cv::InterpolationFlags::INTER_LINEAR);

    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);

    // Initialise ORB detector and detect first frame
    Ptr<FeatureDetector> detector = ORB::create();
    detector->detect(old_gray, kp0);
    KeyPoint::convert(kp0, p0);

    // Shi-Tomasi Detector
    //goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
    
    // Create a mask image for drawing purposes
    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());

    // Initialise with origin
    double x_global = 0, y_global = 0, angle = 0;

#ifdef VID_OUT
    VideoWriter vid_out("output.avi", cv::VideoWriter::fourcc('M','J','P','G'), 30, old_frame.size());
#endif

    int count = 0;

    int center_x = old_frame.size().width/2;
    int center_y = old_frame.size().height/2;
    double x_orth, y_orth;

    while(true){
        count ++;
        capture >> frame;
        //cv::remap(distorted,frame, map1,map2,cv::InterpolationFlags::INTER_LINEAR);
        if (frame.empty())
        break;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        
        // Rerun ORB detector every 5 frames and clear mask
        if (count % 10 == 0){
            detector->detect(frame_gray, kp0);
            KeyPoint::convert(kp0, p0);
            mask = Mat::zeros(old_frame.size(), old_frame.type());
        }

        // calculate optical flow
        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15,15), 2, criteria);
        vector<Point2f> good_new;

        // Initialise 
        Eigen::VectorXf b(2*kp0.size());
        Eigen::VectorXf x(3);
        typedef Eigen::Triplet<float> T;
        std::vector<T> tripletList;
        tripletList.reserve(4*kp0.size());
        int j = 0;

        for(uint i = 0; i < kp0.size(); i++)
        {
            // Select good points
            if(status[i] == 1 ) {
                good_new.push_back(p1[i]);
                // draw the tracks
                line(mask,p1[i], p0[i], colors[i], 2);
                //circle(frame, p1[i], 5, colors[i], -1);

                // Orthogonal Components. Assume rotation about center
                double disp_x = p0[i].x - center_x;
                double disp_y = frame.size().height - p0[i].y - center_y;
                x_orth = -disp_y;
                y_orth = disp_x;

                // Assemble A and B matrix
                b[j]=p1[i].x-p0[i].x;
                b[j+1]=p0[i].y-p1[i].y;

                tripletList.push_back(T(j,0,1));
                tripletList.push_back(T(j,2,x_orth));
                tripletList.push_back(T(j+1,1,1));
                tripletList.push_back(T(j+1,2,y_orth));
                
                j += 2;
            }
        }
        
        b.conservativeResize(j);
        //tripletList.resize(j);
        Eigen::SparseMatrix<float> A(j,3);
        A.setFromTriplets(tripletList.begin(), tripletList.end());
        
        // Invert Matrix here
        Eigen::SparseQR<Eigen::SparseMatrix<float>,Eigen::COLAMDOrdering<int>> solver;
        
        solver.compute(A);
        if(solver.info()!=Eigen::Success) {
            // decomposition failed
            cout << "Decomposition failed" << endl;
        }

        x = solver.solve(b);
        if(solver.info()!=Eigen::Success) {
            // solving failed
            cout << "Solving failed!" << endl;
        }

        y_global -= (sin(angle) * x[0] + cos(angle) * x[1]);
        x_global -= (cos(angle) * x[0] - sin(angle) * x[1]);
        angle -= x[2];

        cout << "Current Location: X = " << x_global <<  ", Y = " << y_global << ", Angle = " << angle << endl; 

        Mat img;
        add(frame, mask, img);
        cv::putText(img, 
            "Current Location: X = " + to_string(x_global) + ", Y = " + to_string(y_global) + ", Angle = " + to_string(angle),
            cv::Point(5,20), // Coordinates (Bottom-left corner of the text string in the image)
            cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
            1.0, // Scale. 2.0 = 2x bigger
            cv::Scalar(255,255,255), // BGR Color
            1, // Line Thickness (Optional)
            cv:: LINE_AA);
        circle(img, Point(-x_global+frame.size().width/2,y_global+frame.size().height/2), 5, CV_RGB(0,0,0) , -1);
        imshow("Frame", img);

#ifdef VID_OUT
        vid_out.write(img);
#endif
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
        // Now update the previous frame and previous points
        old_gray = frame_gray.clone();
        p0 = good_new;
    }
#ifdef VID_OUT    
    vid_out.release();
#endif
}