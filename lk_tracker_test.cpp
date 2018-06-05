#include "lk_tracker.h"

int main()
{
    cv::VideoCapture cap(0);

    cv::Mat frame;

    int cnt = 20;
    while(cnt --)
    {
        cap >> frame;
    }

    int tracker_num = 2;
    std::vector<cv::Rect> rois;
    for(int i = 0; i < tracker_num; i ++)
    {
        cv::Rect roi;
        roi = cv::selectROI("tracker", frame);
        rois.push_back(roi);
    }
 
    auto tm_ptr = new TrackerManager(frame, rois);

    while(1)
    {
        cap >> frame;
        if(frame.empty())
        {
            break;
        }

        cv::Mat img_show = frame.clone();

        tm_ptr->updateTrackersWithNewFrame(frame);
        auto recs = tm_ptr->getAllBox();
        auto pts = tm_ptr->getAllPoints();
        for(auto rec:recs)
        {
            cv::rectangle(img_show, rec, cv::Scalar(0, 240, 0));
        }
        for(auto pt:pts)
        {
            cv::circle(img_show, pt, 10, cv::Scalar(0, 240, 0), 1);
        }
        
        cv::imshow("tracker", img_show);

        if(cv::waitKey(1) == 27)
        {
            break;
        }
    }

    return 0;
}
