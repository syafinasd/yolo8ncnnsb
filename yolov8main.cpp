#include "yoloV8.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>

YoloV8 yolov8;
int target_size = 320; // Choose your target size, must be divisible by 32.

int main(int argc, char** argv)
{
    if (argc != 1)
    {
        fprintf(stderr, "Usage: %s\n", argv[0]);
        return -1;
    }

    cv::VideoCapture cap(0); // Open the default camera (0) or another camera (e.g., 1 for external camera)
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera.\n";
        return -1;
    }

    yolov8.load(target_size); // Load model (once)

    cv::Mat frame;
    std::vector<Object> objects;

    while (true)
    {
        cap >> frame; // Capture frame-by-frame
        if (frame.empty())
        {
            std::cerr << "Error: Blank frame grabbed\n";
            break;
        }

        yolov8.detect(frame, objects); // Detect objects in the frame
        yolov8.draw(frame, objects);   // Draw bounding boxes and labels

        cv::imshow("Object Detection", frame);

        if (cv::waitKey(1) == 27) // Exit loop if ESC is pressed
            break;
    }

    cap.release(); // Release the camera
    cv::destroyAllWindows(); // Close all OpenCV windows

    return 0;
}
