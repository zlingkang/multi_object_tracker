#include "object_detection.h"

ObjectDetection::ObjectDetection()
{
    face_cascade_.load("../haarcascade_frontalface_alt.xml");
}

std::vector<cv::Rect> ObjectDetection::detectObject(const cv::Mat& _frame)
{
    std::vector<cv::Rect> boxes;
    face_cascade_.detectMultiScale(_frame, boxes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30,30)); 
    return boxes;
}
