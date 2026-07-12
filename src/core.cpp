#include "pose.hpp"
#include "hands.hpp"

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <std_msgs/UInt32.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;

Intrinsics intrinsics;
Poser poser;
Handser handser;

// Pose publishers
ros::Publisher pose_pub;
ros::Publisher pose_annot;
ros::Publisher pose_vis;
ros::Publisher pose_kpt_vis;

// Hand publishers
ros::Publisher hands_pub;
ros::Publisher hands_annot;
ros::Publisher hands_vis;

// Target subscription
ros::Subscriber target_sub;
ros::Publisher  target_ack;
uint32_t target = 0;


// Target callback
void target_callback(const std_msgs::UInt32::ConstPtr& msg) {
    target = msg->data;
}


// "Main loop"
void callback(const sensor_msgs::Image::ConstPtr& rgb_msg,
              const sensor_msgs::Image::ConstPtr& depth_msg) {
    // 1: Sanity checks on frames
    if ( intrinsics.empty() ) {
        ROS_INFO_THROTTLE(2.0, "person_follower/pose: waiting for camera intrinsics");
        return;
    }

    ImagePtrStruct data;

    try {
        data.rgb = cv_bridge::toCvShare(rgb_msg, sensor_msgs::image_encodings::BGR8);
        data.depth = cv_bridge::toCvShare(depth_msg, depth_msg->encoding);
    } catch ( const cv_bridge::Exception& e ) {
        ROS_ERROR_THROTTLE(2.0, "person_follower/pose: cv_bridge exception '%s'", e.what());
        return;
    }

    // 2: Run YOLO for pose estimation
    PoseArrayMsg poses = poser.detect(data, intrinsics, pose_annot);
    pose_pub.publish(poses);

    if ( pose_vis.getNumSubscribers() )
        pose_vis.publish(poser.pose_center_markers(poses));

    if ( pose_kpt_vis.getNumSubscribers() )
        pose_kpt_vis.publish(poser.keypoint_markers(data, intrinsics, poses));

    // 3: Run MediaPipe hands
    std_msgs::UInt32 target_ack_msg;
    target_ack_msg.data = target;
    target_ack.publish(target_ack_msg);

    PoseArrayMsg hands = handser.update(data, intrinsics, poses, target, poser.keypoint_conf, hands_annot);
    hands_pub.publish(hands);

    if ( hands_vis.getNumSubscribers() )
        hands_vis.publish(handser.landmarker_markers(hands));
}

// Entry point
int main(int argc, char** argv) {
    ros::init(argc, argv, "person_follower");

    ros::NodeHandle nh;
    poser.init();
    handser.init();
    intrinsics.init();

    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/color/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/depth/image_raw", 1);

    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), rgb_sub, depth_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2));

    pose_pub     = nh.advertise<PoseArrayMsg>("/person_follower/pose", 1);
    pose_annot   = nh.advertise<sensor_msgs::Image>("/person_follower/pose/image_raw", 1);
    pose_vis     = nh.advertise<visualization_msgs::MarkerArray>("/person_follower/pose/markers", 1);
    pose_kpt_vis = nh.advertise<visualization_msgs::MarkerArray>("/person_follower/pose/keypoint_markers", 1);

    hands_pub   = nh.advertise<PoseArrayMsg>("/person_follower/hands", 1);
    hands_annot = nh.advertise<sensor_msgs::Image>("/person_follower/hands/image_raw", 1);
    hands_vis   = nh.advertise<visualization_msgs::MarkerArray>("/person_follower/hands/markers", 1);

    target_sub = nh.subscribe("/person_follower/target", 1, target_callback);
    target_ack = nh.advertise<std_msgs::UInt32>("/person_follower/target/ack", 1);

    ROS_INFO("starting to spin");
    ros::spin();
}