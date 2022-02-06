#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/opencv.hpp"

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
    Mat old_frame, old_gray, frame, frame_gray;
    vector<KeyPoint> kp0;
    vector<Point2f> p0, p1;

    // Initial frame
    capture >> old_frame;
    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);

    // Initialise ORB detector and detect first frame
    Ptr<FeatureDetector> detector = ORB::create();
    detector->detect(old_gray, kp0);
    KeyPoint::convert(kp0, p0);
    
    // Shi-Tomasi Detector
    //goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
    
    // Create a mask image for drawing purposes
    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());

    double x = 0, y = 0;

    VideoWriter vid_out("outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), 30, old_frame.size());

    int count = 0;
    while(true){
        count ++;
        vector<Point2f> diff;
        capture >> frame;
        if (frame.empty())
        break;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        
        // Rerun ORB detector every 5 frames
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
        for(uint i = 0; i < kp0.size(); i++)
        {
            // Select good points
            if(status[i] == 1 ) {
                good_new.push_back(p1[i]);
                // draw the tracks
                line(mask,p1[i], p0[i], colors[i], 2);
                //circle(frame, p1[i], 5, colors[i], -1);
                diff.push_back(p1[i]-p0[i]);
            }
        }

        //Process diff vector here
        double sum_x = 0;
        double sum_y = 0;
        for(uint i = 0; i < diff.size(); i++)
        {
            sum_x += diff[i].x;
            sum_y += diff[i].y;
        }
        x -= sum_x / (diff.size() + 1);
        y += sum_y / (diff.size() + 1);

        cout << "Current Location: " << x <<  ", " << y << endl; 

        Mat img;
        add(frame, mask, img);
        imshow("Frame", img);
        cv::putText(frame, 
            "Current Location: " + to_string(x) + " " + to_string(y),
            cv::Point(5,20), // Coordinates (Bottom-left corner of the text string in the image)
            cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
            1.0, // Scale. 2.0 = 2x bigger
            cv::Scalar(255,255,255), // BGR Color
            1, // Line Thickness (Optional)
            cv:: LINE_AA);
        circle(frame, Point(-x+1280/2,y+720/2), 5, CV_RGB(0,0,0) , -1);
        vid_out.write(frame);
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
        // Now update the previous frame and previous points
        old_gray = frame_gray.clone();
        p0 = good_new;
    }
    vid_out.release();
}