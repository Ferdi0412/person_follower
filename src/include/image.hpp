#ifndef IMAGE_PTR_STRUCT_HPP_
#define IMAGE_PTR_STRUCT_HPP_

#include <ros/ros.h>

#include <sensor_msgs/CameraInfo.h>

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

struct Intrinsics {
    float fx{0.0f};
    float fy{0.0f};
    float cx{0.0f};
    float cy{0.0f};

    void init() {
        ros::NodeHandle nh;
        sub = nh.subscribe("/camera/depth/camera_info", 1, &Intrinsics::callback, this);
    }

    bool empty() const {
        return fx == 0.0f || fy == 0.0f;
    }

    /// @brief Get the real-world coordinate of a depth and position in frame
    cv::Point3f world(float u, float v, float z) const {
        return cv::Point3f(
            (u - cx) * z / fx,
            (v - cy) * z / fy,
            z
        );
    } 

    private:
        ros::Subscriber sub;

        void callback(const sensor_msgs::CameraInfo::ConstPtr& msg) {
            fx = static_cast<float>(msg->K[0]);
            fy = static_cast<float>(msg->K[4]);
            cx = static_cast<float>(msg->K[2]);
            cy = static_cast<float>(msg->K[5]);

            sub.shutdown();
        }
};

struct ImagePtrStruct {
    cv_bridge::CvImageConstPtr rgb;
    cv_bridge::CvImageConstPtr depth;

    /// @brief Get the appropriate header
    std_msgs::Header header() const {
        return depth->header;
    }

    /// @brief Get the depth at a point in the depth frame
    float z(int u, int v) const {
        return depth->image.at<uint16_t>(v, u) * 0.001;
    }

    /// @brief Get a clone of the rgb image for annotating
    cv_bridge::CvImagePtr clone_rgb() const {
        return boost::make_shared<cv_bridge::CvImage>(
            rgb->header,
            rgb->encoding,
            rgb->image.clone()
        );
    }
};

#endif 