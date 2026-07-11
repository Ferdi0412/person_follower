#include "IntrinsicsSubscriber.hpp"
#include "YoloWrapper.hpp"

#include <ros/ros.h>

#include <person_follower_msgs/PoseArray.h>
#include <person_follower_msgs/Pose.h>
#include <person_follower_msgs/Keypoint.h>

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;

using Clock = std::chrono::high_resolution_clock;
using PoseArrayMsg = person_follower_msgs::PoseArray;
using PoseMsg      = person_follower_msgs::Pose;
using KeypointMsg  = person_follower_msgs::Keypoint;


IntrinsicsSubscriber intrinsics;
YoloWrapper yolo;

// Pose publishers
ros::Publisher pose_pub;
ros::Publisher pose_annot;
ros::Publisher pose_vis;
ros::Publisher pose_kpt_vis;



/// Main loop
void callback(const sensor_msgs::Image::ConstPtr& rgb_msg,
              const sensor_msgs::Image::ConstPtr& depth_msg) {
    if ( intrinsics.empty() ) {
        ROS_INFO_THROTTLE(2.0, "person_follower/pose: waiting for camera intrinsics");
        return;
    }

    cv_bridge::CvImageConstPtr rgb;
    cv_bridge::CvImageConstPtr depth;

    try {
        rgb = cv_bridge::toCvShare(rgb_msg, sensor_msgs::image_encodings::BGR8);
        depth = cv_bridge::toCvShare(depth_msg, depth_msg->encoding);
    } catch ( const cv_bridge::Exception& e ) {
        ROS_ERROR_THROTTLE(2.0, "person_follower/pose: cv_bridge exception '%s'", e.what());
        return;
    }

    cv_bridge::CvImagePtr pose_frame;
    if ( pose_annot.getNumSubscribers() ) {
        pose_frame = boost::make_shared<cv_bridge::CvImage>(
            rgb->header,
            rgb->encoding,
            rgb->image.clone()
        );
    }
    
    PoseArrayMsg pose_msg;
    yolo.process(rgb, depth, intrinsics, pose_msg, pose_frame);

    pose_pub.publish(pose_msg);

    if ( pose_frame )
        pose_annot.publish(pose_frame->toImageMsg());

    if ( pose_vis )
        pose_vis.publish(yolo.pose_visualization(pose_msg));

    if ( pose_kpt_vis )
        pose_kpt_vis.publish(yolo.joint_visualization(pose_msg, depth, intrinsics));
}


/// Entry point
int main(int argc, char** argv) {
    ros::init(argc, argv, "pose");
    ros::NodeHandle nh;
    ros::NodeHandle args("~");

    // 1: Set up models
    yolo.init();

    // 2: Subscribe to cameras
    std::string camera_topic = args.param<std::string>("/camera", "/camera");
    std::string rgb_topic    = args.param<std::string>("/rgb", camera_topic + "/color");
    std::string depth_topic  = args.param<std::string>("/depth", camera_topic + "/depth");

    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, rgb_topic + "/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, depth_topic + "/image_raw", 1);

    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), rgb_sub, depth_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2));

    // 4: Set up publishers
    pose_pub     = nh.advertise<PoseArrayMsg>("/person_follower/pose", 1);
    pose_annot   = nh.advertise<sensor_msgs::Image>("/person_follower/pose/image_raw", 1);
    pose_vis     = nh.advertise<visualization_msgs::MarkerArray>("/person_follower/pose/markers", 1);
    pose_kpt_vis = nh.advertise<visualization_msgs::MarkerArray>("/person_follower/pose/joint_markers", 1);

    // 5: Subscribe to camera intrinsics
    intrinsics.init(depth_topic + "/camera_info");

    // 6: Start with rate-limiting if appropriate
    float rate = args.param<float>("/rate", 0.0f);

    if ( rate <= 0.0f ) {
        ROS_INFO("person_follower/pose: starting");
        ros::spin();
    }

    else {
        ROS_INFO("person_follower/pose: starting with target rate %.2f Hz", rate);
        ros::Rate loop_rate(rate);
        while ( ros::ok() ) {
            ros::spinOnce();

            if ( !loop_rate.sleep() )
                ROS_WARN_THROTTLE(2.0, "person_follower/pose: failed to maintain target %.2f Hz", rate);
        }
    }
}