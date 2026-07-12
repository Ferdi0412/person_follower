#include "person_follower/reid/osnet.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>

using person_follower::reid::OSNet;
using person_follower::reid::similarity;

cv::Rect centered_roi(const cv::Mat& img, int height) {
    int width = height / 2;
    cv::Rect roi(
        img.cols / 2 - width / 2,
        img.rows / 2 - height / 2,
        width,
        height
    );
    return roi & cv::Rect(0, 0, img.cols, img.rows); // clamp to bounds
}

int main() {
    cv::VideoCapture cap(6);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera\n";
        return 1;
    }

    OSNet model(
        "/home/ferdi/Documents/catkin_ws/src/person_follower/models/osnet_x0_5.onnx",
        "cpu",
        1
    );

    std::vector<float> ref;
    bool have_ref = false;

    cv::Mat img = cv::imread("frame.jpg");
    if (img.empty()) {
        std::cerr << "frame.jpg not found — press 's' during capture to save a reference frame first.\n";
    } else {
        int height = 256 * 3;
        cv::Rect roi = centered_roi(img, height);
        if (roi.width > 0 && roi.height > 0) {
            ref = model.embed(img(roi));
            have_ref = true;
        } else {
            std::cerr << "frame.jpg too small for requested crop size.\n";
        }
    }

    while (cap.isOpened()) {
        cv::Mat frame;
        if (!cap.read(frame))
            break;

        int height = 256 * 3;
        cv::Rect roi = centered_roi(frame, height);

        if (roi.width > 0 && roi.height > 0) {
            auto res = model.embed(frame(roi));
            if (have_ref)
                std::cout << similarity(res, ref) << std::endl;

            cv::rectangle(frame, roi, cv::Scalar(250, 50, 50), 1);
        }

        cv::imshow("frame", frame);

        int key = cv::waitKey(1);
        if (key == 'q')
            break;
        if (key == 's' && roi.width > 0 && roi.height > 0)
            cv::imwrite("frame.jpg", frame(roi).clone());
    }

    return 0;
}