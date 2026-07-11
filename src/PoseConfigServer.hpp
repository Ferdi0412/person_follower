#ifndef POSE_CONFIG_SERVER_HPP_
#define POSE_CONFIG_SERVER_HPP_

#include <ros/ros.h>

#include <dynamic_reconfigure/server.h>
#include <person_follower/PoseConfig.h>

struct PoseConfigServer {
    const std::string name;
    float detection_conf;
    float detection_iou;
    float conf;
    float roi;
    bool  log_infer_time;
    bool  log_full_time;

    /// Initialize and start config server
    void init(ros::NodeHandle nh) {
        if ( server )
            return;

        server = std::make_unique<dynamic_reconfigure::Server<person_follower::PoseConfig>>(nh);
        f = boost::bind(&PoseConfigServer::reconfigure, this, _1, _2);
        server->setCallback(f);
    }

    /// Construct with defaults in case of `0.0f`
    PoseConfigServer(
        const std::string& name_,
        float default_detection_conf = 0.5,
        float default_detection_iou  = 0.5,
        float default_conf          = 0.5,
        float default_roi            = 0.5
    ) : name(name_),
        d_detection_conf(default_detection_conf),
        d_detection_iou(default_detection_iou),
        d_conf(default_conf),
        d_roi(default_roi),
        detection_conf(default_detection_conf),
        detection_iou(default_detection_iou),
        conf(default_conf),
        roi(default_roi),
        log_infer_time(false),
        log_full_time(false) {}

    private:
        const float d_detection_conf;
        const float d_detection_iou;
        const float d_conf;
        const float d_roi;

        std::unique_ptr<dynamic_reconfigure::Server<person_follower::PoseConfig>> server;
        dynamic_reconfigure::Server<person_follower::PoseConfig>::CallbackType f;

        /// Callback to change config
        void reconfigure(person_follower::PoseConfig& config, uint32_t level) {
            set(detection_conf, config.detection_conf, d_detection_conf);
            set(detection_iou, config.detection_iou, d_detection_iou);
            set(conf, config.conf, d_conf);
            set(roi, config.roi, d_roi);
            log_infer_time = config.log_infer_time;
            log_full_time = config.log_full_time;

            if ( level & 1 )
                ROS_INFO(
                    "%s - new detection criteria - conf:=%.2f | iou:=%.2f",
                    name.c_str(),
                    detection_conf,
                    detection_iou
                );
            if ( level & 2 )
                ROS_INFO(
                    "%s - new per-instance criteria - conf:=%.2f | roi:=%.2f",
                    name.c_str(),
                    conf,
                    roi
                );
        }

        /// Sets the fallback value if cfg is 0
        void set(float& field, double& cfg, float fallback) {
            if ( cfg <= 0.0f ) {
                field = fallback;
                cfg = fallback;
            }
            else
                field = cfg;
        }
};

#endif // POSE_CONFIG_SERVER_HPP_