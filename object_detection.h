#ifndef OBJECT_DETECTION_H
#define OBJECT_DETECTION_H

#include "opencv2/core/utility.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <vector>

class ObjectDetection
{
    public:
        ObjectDetection();
        std::vector<cv::Rect> detectObject(const cv::Mat& _frame);
    private:
        cv::CascadeClassifier face_cascade_;
};

#endif
