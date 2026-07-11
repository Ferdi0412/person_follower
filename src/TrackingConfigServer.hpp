#ifndef PERSON_FOLLOWER_TRACKING_CONFIG_SERVER_HPP_
#define PERSON_FOLLOWER_TRACKING_CONFIG_SERVER_HPP_

#include <ros/ros.h>

#include <dynamic_reconfigure/server.h>
#include <person_follower/TrackingConfig.h>

struct TrackingConfigServer {
    const std::string name;
    float iou;
    int   min_hits;
    int   max_age;

    // Initialize and start config server
    void init(ros::NodeHandle nh) {
        if ( server )
            return;

        server = std::make_unique<dynamic_reconfigure::Server<person_follower::TrackingConfig>>(nh);
        f = boost::bind(&TrackingConfigServer::reconfigure, this, _1, _2);
        server->setCallback(f);
    }

    /// Construct with defaults in case of 0s
    TrackingConfigServer(
        const std::string& name_,
        float default_iou      = 0.3f,
        int   default_min_hits = 3,
        int   default_max_age  = 3
    ) : name(name_),
        d_iou(default_iou),
        d_max_age(default_max_age),
        d_min_hits(default_min_hits),
        iou(default_iou),
        max_age(default_max_age),
        min_hits(default_min_hits) {}

    private:
        const float d_iou;
        const int   d_min_hits;
        const int   d_max_age;

        std::unique_ptr<dynamic_reconfigure::Server<person_follower::TrackingConfig>> server;
        dynamic_reconfigure::Server<person_follower::TrackingConfig>::CallbackType f;

        /// Callback to change config
        void reconfigure(person_follower::TrackingConfig& config, uint32_t level) {
            set(iou, config.iou, d_iou);
            set(min_hits, config.min_hits, d_min_hits);
            set(max_age, config.max_age, d_max_age);

            ROS_INFO(
                "%s - new tracking config - iou:=%.2f | min_hits:=%d | max_age:=%d",
                name.c_str(),
                iou,
                min_hits,
                max_age
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

        /// Sets the fallback value if cfg is 0
        void set(int& field, int& cfg, int fallback) {
            if ( cfg <= 1 ) {
                field = fallback;
                cfg = fallback;
            }
            else
                field = cfg;
        }
};

#endif // PERSON_FOLLOWER_TRACKING_CONFIG_SERVER_HPP_