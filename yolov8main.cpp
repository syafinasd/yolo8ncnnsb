#include "yoloV8.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>

YoloV8 yolov8;
int target_size = 320; // Smaller size for better performance on RPi4

int main(int argc, char** argv)
{
    if (argc != 1)
    {
        fprintf(stderr, "Usage: %s\n", argv[0]);
        return -1;
    }

    cv::VideoCapture cap(0); // Open the default camera
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera.\n";
        return -1;
    }

    yolov8.load(target_size); // Load model (once)

    cv::Mat frame;
    std::vector<Object> objects;
    int64 start, end;
    double fps, inference_time;

    while (true)
    {
        cap >> frame; // Capture frame-by-frame
        if (frame.empty())
        {
            std::cerr << "Error: Blank frame grabbed\n";
            break;
        }

        start = cv::getTickCount();
        yolov8.detect(frame, objects); // Detect objects in the frame
        end = cv::getTickCount();

        inference_time = (end - start) / cv::getTickFrequency();
        fps = 1.0 / inference_time;

        yolov8.draw(frame, objects);   // Draw bounding boxes and labels

        // Display FPS and inference time on the frame
        std::string fps_text = cv::format("FPS: %.2f", fps);
        std::string time_text = cv::format("Inference time: %.2f ms", inference_time * 1000);

        cv::putText(frame, fps_text, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, time_text, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Object Detection", frame);

        if (cv::waitKey(1) == 27) // Exit loop if ESC is pressed
            break;
    }

    cap.release(); // Release the camera
    cv::destroyAllWindows(); // Close all OpenCV windows

    return 0;
}
