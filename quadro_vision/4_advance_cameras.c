#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    // initialize camera capture
    VideoCapture cap1(0), cap2(1), cap3(2), cap4(3);

    // set the resolution for all cameras
    cap1.set(CAP_PROP_FRAME_WIDTH, 640);
    cap1.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap2.set(CAP_PROP_FRAME_WIDTH, 640);
    cap2.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap3.set(CAP_PROP_FRAME_WIDTH, 640);
    cap3.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap4.set(CAP_PROP_FRAME_WIDTH, 640);
    cap4.set(CAP_PROP_FRAME_HEIGHT, 480);

    // initialize stereo block matching algorithms
    Ptr<StereoBM> stereo12 = StereoBM::create(16, 15);
    Ptr<StereoBM> stereo13 = StereoBM::create(16, 15);
    Ptr<StereoBM> stereo14 = StereoBM::create(16, 15);
    Ptr<StereoBM> stereo23 = StereoBM::create(16, 15);
    Ptr<StereoBM> stereo24 = StereoBM::create(16, 15);
    Ptr<StereoBM> stereo34 = StereoBM::create(16, 15);

    // camera parameters for distance calculation
    float focal_length = 3.6;  // in mm
    float sensor_width = 6.17;  // in mm
    int baseline12 = 75;  // in mm
    int baseline13 = 100;  // in mm
    int baseline14 = 50;  // in mm
    int baseline23 = 75;  // in mm
    int baseline24 = 50;  // in mm
    int baseline34 = 75;  // in mm

    // line scanning parameters
    int horizon = 200;  // horizon line for line scanning
    int step = 10;  // pixel step size for line scanning

    while (true) {
        // read frames from all cameras
        Mat frame1, frame2, frame3, frame4;
        cap1.read(frame1);
        cap2.read(frame2);
        cap3.read(frame3);
        cap4.read(frame4);

        // convert frames to grayscale
        Mat gray1, gray2, gray3, gray4;
        cvtColor(frame1, gray1, COLOR_BGR2GRAY);
        cvtColor(frame2, gray2, COLOR_BGR2GRAY);
        cvtColor(frame3, gray3, COLOR_BGR2GRAY);
        cvtColor(frame4, gray4, COLOR_BGR2GRAY);

        // compute disparity maps
        Mat disparity12, disparity13, disparity14, disparity23, disparity24, disparity34;
        stereo12->compute(gray1, gray2, disparity12);
        stereo13->compute(gray1, gray3, disparity13);
        stereo14->compute(gray1, gray4, disparity14);
        stereo23->compute(gray2, gray3, disparity23);
        stereo24->compute(gray2, gray4, disparity24);
        stereo34->compute(gray3, gray4, disparity34);

       // apply thresholding to disparity maps
Mat thresh12, thresh13, thresh14, thresh23, thresh24, thresh34;
threshold(disparity12, thresh12, 0, 255, THRESH_BINARY);
threshold(disparity13, thresh13, 0, 255, THRESH_BINARY);
threshold(disparity14, thresh14, 0, 255, THRESH_BINARY);
threshold(disparity23, thresh23, 0, 255, THRESH_BINARY);
threshold(disparity24, thresh24, 0, 255, THRESH_BINARY);
threshold(disparity34, thresh34, 0, 255, THRESH_BINARY);

// apply morphological opening to remove noise
Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
Mat opening12, opening13, opening14, opening23, opening24, opening34;
morphologyEx(thresh12, opening12, MORPH_OPEN, kernel);
morphologyEx(thresh13, opening13, MORPH_OPEN, kernel);
morphologyEx(thresh14, opening14, MORPH_OPEN, kernel);
morphologyEx(thresh23, opening23, MORPH_OPEN, kernel);
morphologyEx(thresh24, opening24, MORPH_OPEN, kernel);
morphologyEx(thresh34, opening34, MORPH_OPEN, kernel);

// scan the horizon line for lines in all camera pairs
for (int i = 0; i < gray1.cols / step; i++) {
    int x1 = i * step, y1 = horizon;
    int x2 = (i + 1) * step, y2 = horizon;

        if (!opening12.empty()) {
        std::vector<Vec4i> lines;
        HoughLinesP(opening12(Rect(x1, y1, x2 - x1, y2 - y1)), lines, 1, CV_PI / 180, 100, 50, 10);

        for (size_t j = 0; j < lines.size(); j++) {
            Vec4i l = lines[j];
            int x1_ = l[0];
            int y1_ = l[1];
            int x2_ = l[2];
            int y2_ = l[3];

            double depth12 = (focal_length * baseline12) / (disparity12.at<short>(y1_ + y1, x1_ + x1) + 0.0000001);
            line(frame1, Point(x1 + x1_, y1 + y1_), Point(x1 + x2_, y1 + y2_), Scalar(0, 255, 0), 2);
        }
    }

    if (!opening13.empty()) {
        std::vector<Vec4i> lines;
        HoughLinesP(opening13(Rect(x1, y1, x2 - x1, y2 - y1)), lines, 1, CV_PI / 180, 100, 50, 10);

        for (size_t j = 0; j < lines.size(); j++) {
            Vec4i l = lines[j];
            int x1_ = l[0];
            int y1_ = l[1];
            int x2_ = l[2];
            int y2_ = l[3];

            double depth13 = (focal_length * baseline13) / (disparity13.at<short>(y1_ + y1, x1_ + x1) + 0.0000001);
            line(frame1, Point(x1 + x1_, y1 + y1_), Point(x1 + x2_, y1 + y2_), Scalar(0, 255, 0), 2);
        }
    }

    if (!opening14.empty()) {
        std::vector<Vec4i> lines;
        HoughLinesP(opening14(Rect(x1, y1, x2 - x1, y2 - y1)), lines, 1, CV_PI / 180, 100, 50, 10);

        for (size_t j = 0; j < lines.size(); j++) {
            Vec4i l = lines[j];
            int x1_ = l[0];
            int y1_ = l[1];
            int x2_ = l[2];
            int y2_ = l[3];

            double depth14 = (focal_length * baseline14) / (disparity14.at<short>(y1_ + y1, x1_ + x1) + 0.0000001);
            line(frame1, Point(x1 + x1_, y1 + y1_), Point(x1 + x2_, y1 + y2_), Scalar(0, 255, 0), 2);
        }
    }

   if (!opening23.empty()) {
std::vector<Vec4i> lines;
HoughLinesP(opening23(Rect(x1, y1, x2 - x1, y2 - y1)), lines, 1, CV_PI / 180, 100, 50, 10);
        for (size_t j = 0; j < lines.size(); j++) {
            Vec4i l = lines[j];
            int x1_ = l[0];
            int y1_ = l[1];
            int x2_ = l[2];
            int y2_ = l[3];

            double depth23 = (focal_length * baseline23) / (disparity23.at<short>(y1_ + y1, x1_ + x1) + 0.0000001);
            line(frame2, Point(x1 + x1_, y1 + y1_), Point(x1 + x2_, y1 + y2_), Scalar(0, 255, 0), 2);
        }
    }

    if (!opening24.empty()) {
        std::vector<Vec4i> lines;
        HoughLinesP(opening24(Rect(x1, y1, x2 - x1, y2 - y1)), lines, 1, CV_PI / 180, 100, 50, 10);

        for (size_t j = 0; j < lines.size(); j++) {
            Vec4i l = lines[j];
            int x1_ = l[0];
            int y1_ = l[1];
            int x2_ = l[2];
            int y2_ = l[3];

            double depth24 = (focal_length * baseline24) / (disparity24.at<short>(y1_ + y1, x1_ + x1) + 0.0000001);
            line(frame2, Point(x1 + x1_, y1 + y1_), Point(x1 + x2_, y1 + y2_), Scalar(0, 255, 0), 2);
        }
    }

    if (!opening34.empty()) {
        std::vector<Vec4i> lines;
        HoughLinesP(opening34(Rect(x1, y1, x2 - x1, y2 - y1)), lines, 1, CV_PI / 180, 100, 50, 10);

        for (size_t j = 0; j < lines.size(); j++) {
            Vec4i l = lines[j];
            int x1_ = l[0];
            int y1_ = l[1];
            int x2_ = l[2];
            int y2_ = l[3];

            double depth34 = (focal_length * baseline34) / (disparity34.at<short>(y1_ + y1, x1_ + x1) + 0.0000001);
            line(frame3, Point(x1 + x1_, y1 + y1_), Point(x1 + x2_, y1 + y2_), Scalar(0, 255, 0), 2);
        }
    }
}
// calculate 3D coordinates of the point of intersection of the four lines
Mat a = (Mat_<double>(4, 3) << 0, 0, depth12, gray1.cols, 0, depth14, gray2.cols, 0, depth24, gray3.cols, 0, depth34);
Mat A = a.t() * a;
Mat B = (Mat_<double>(4, 1) << a.at<double>(0, 2) * a.at<double>(1, 1) * a.at<double>(2, 0) * a.at<double>(3, 1) -
                                a.at<double>(0, 1) * a.at<double>(1, 2) * a.at<double>(2, 0) * a.at<double>(3, 1) -
                                a.at<double>(0, 2) * a.at<double>(1, 0) * a.at<double>(2, 1) * a.at<double>(3, 1) +
                                a.at<double>(0, 0) * a.at<double>(1, 2) * a.at<double>(2, 1) * a.at<double>(3, 1) +
                                a.at<double>(0, 1) * a.at<double>(1, 0) * a.at<double>(2, 2) * a.at<double>(3, 1) -
                                a.at<double>(0, 0) * a.at<double>(1, 1) * a.at<double>(2, 2) * a.at<double>(3, 1) -
                                a.at<double>(0, 2) * a.at<double>(1, 1) * a.at<double>(2, 0) * a.at<double>(3, 3) +
                                a.at<double>(0, 1) * a.at<double>(1, 2) * a.at<double>(2, 0) * a.at<double>(3, 3) +
                                a.at<double>(0, 2) * a.at<double>(1, 0) * a.at<double>(2, 1) * a.at<double>(3, 3) -
                                a.at<double>(0, 0) * a.at<double>(1, 2) * a.at<double>(2, 1) * a.at<double>(3, 3) -
                                a.at<double>(0, 1) * a.at<double>(1, 0) * a.at<double>(2, 2) * a.at<double>(3, 3) +
                                a.at<double>(0, 0) * a.at<double>(1, 1) * a.at<double>(2, 2) * a.at<double>(3, 3));

Mat point_3d;
solve(A, B, point_3d);

// display 3D point on the image
Point3f pt(point_3d.at<double>(0), point_3d.at<double>(1), point_3d.at<double>(2));
drawAxis(frame1, camera_matrix, distortion_coeffs, rvecs12, tvecs12, 0.3f);
drawPoint(frame1, camera_matrix, distortion_coeffs, rvecs12, tvecs12, pt, 0.02f);

// display the image with detected 3D point and Hough lines
imshow("3D Point Detection", frame1);

// wait for key press and exit if ESC is pressed
char key = (char)waitKey(10);
if (key == 27)
    break;
Mat X = A.inv() * a.t() * B;
double X_ = X.at<double>(0, 0);
double Y_ = X.at<double>(1, 0);
double Z_ = X.at<double>(2, 0);

// convert 3D point from camera frame to world frame
Mat point_c = (Mat_<double>(4, 1) << X_, Y_, Z_, 1);
Mat point_w = R_w_c.t() * (point_c - t_w_c);

// print out the 3D coordinates of the point in the world frame
cout << "Point coordinates in world frame: (" << point_w.at<double>(0, 0) << ", "
<< point_w.at<double>(1, 0) << ", " << point_w.at<double>(2, 0) << ")" << endl;

// draw circle at the 3D point on the output image
Point center(point_w.at<double>(0, 0), point_w.at<double>(1, 0));
circle(output, center, 5, Scalar(255, 0, 0), -1);
}
}

// display output image
imshow("Output", output);

// press any key to exit
waitKey(0);

return 0;
}

// calculate 3D coordinates of the point of intersection of the four lines
Mat a = (Mat_<double>(4, 3) << 0, 0, depth12, gray1.cols, 0, depth14, gray2.cols, 0, depth24, gray3.cols, 0, depth34);
Mat A = a.t() * a;
Mat B = (Mat_<double>(4, 1) << -depth12 * gray1.rows, -depth14 * gray2.rows, -depth24 * gray3.rows, -depth34 * gray4.rows);
Mat x = A.inv() * (a.t() * B);

// draw point of intersection on the first frame
circle(frame1, Point(x.at<double>(0), x.at<double>(1)), 5, Scalar(0, 255, 0), 2);
// draw point of intersection on the first frame
circle(frame1, Point(x.at<double>(0), x.at<double>(1)), 5, Scalar(0, 255, 0), 2);

// display the frames with detected points and lines
imshow("Frame 1", frame1);
imshow("Frame 2", frame2);
imshow("Frame 3", frame3);
imshow("Frame 4", frame4);

// wait for key press and exit if 'q' is pressed
char c = waitKey(0);
if (c == 'q') {
break;
}

// clear the vectors for next iteration
lines12.clear();
lines13.clear();
lines14.clear();
lines23.clear();
lines24.clear();
lines34.clear();
points.clear();
}
// release the video capture object and destroy all windows
cap.release();
destroyAllWindows();

return 0;
}
