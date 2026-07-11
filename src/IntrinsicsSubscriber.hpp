#ifndef INTRINSICS_SUBSCRIBER_HPP_
#define INTRINSICS_SUBSCRIBER_HPP_

#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>

#include <opencv2/core.hpp>

struct IntrinsicsSubscriber {
    float fx{0.0f};
    float fy{0.0f};
    float cx{0.0f};
    float cy{0.0f};

    /// Set up subcription
    void init(const std::string& topic) {
        ros::NodeHandle nh;
        init(topic, nh);
    }

    /// Set up subscription
    void init(const std::string& topic, ros::NodeHandle& nh) {
        sub = nh.subscribe(topic, 1, &IntrinsicsSubscriber::callback, this);
    }

    /// Check that intrinsics have been received
    bool empty() const {
        return (fx == 0) || (fy == 0);
    }

    /// Get projection from image plane into word
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

#endif // INTRINSICS_SUBSCRIBER_HPP_