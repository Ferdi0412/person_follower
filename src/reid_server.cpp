#include "person_follower/reid/osnet.hpp"

#include <ros/ros.h>

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <mutex>
#include <condition_variable>

#include <person_follower_msgs/EmbedRoi.h>

using person_follower::reid::OSNet;
using person_follower::utils::person_follower_dir;

class EmbeddingNode {
    public:
        EmbeddingNode(ros::NodeHandle& nh) {
            sub = nh.subscribe("/camera/color/image_raw", 1, &EmbeddingNode::callback, this);
            service = nh.advertiseService("get_embedding", &EmbeddingNode::request, this);
            osnet = std::make_unique<OSNet>(person_follower_dir("models/osnet_x0_5.onnx"), "gpu");
        }

    private:
        void callback(const sensor_msgs::ImageConstPtr& msg) {
            std::unique_lock<std::mutex> lock(mutex);

            if ( !pending_request )
                return;

            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
            
            if ( roi.x < 0 || roi.y < 0 || (roi.x + roi.width) > cv_ptr->image.cols || (roi.y + roi.height) > cv_ptr->image.rows ) {
                error_message = "roi outside image limits";
                embedding = std::vector<float>{};
                pending_request = false;
                result_ready = true;
            }

            cv::Mat image;
            if ( roi.x == 0 && roi.y == 0 && roi.width == image.cols && roi.height == image.rows )
                image = cv_ptr->image;

            else
                image = cv_ptr->image(roi).clone();

            embedding = osnet->embed(image);

            pending_request = false;
            result_ready    = true;
            lock.unlock();
            cond.notify_one();
        }

        bool request(person_follower_msgs::EmbedRoi::Request& req,
                     person_follower_msgs::EmbedRoi::Response& resp) {
            std::unique_lock<std::mutex> lock(mutex);

            roi = cv::Rect (
                req.x,
                req.y,
                req.width,
                req.height
            );

            pending_request = true;
            result_ready    = false;

            bool ok = cond.wait_for(lock, std::chrono::milliseconds(500), [this]{ return result_ready; });

            if ( !ok ) {
                pending_request = false;
                resp.error_message = "timeout";
                return true;
            }

            resp.embedding = embedding;
            resp.error_message = error_message;
            return true;
        }

        ros::Subscriber sub;
        ros::ServiceServer service;

        std::mutex mutex;
        std::condition_variable cond;

        bool pending_request{false};
        bool result_ready{false};

        cv::Rect roi;
        std::string error_message;
        std::vector<float> embedding;

        std::unique_ptr<OSNet> osnet;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "person_follower_embed_service");
    ros::NodeHandle nh;
    ros::AsyncSpinner spinner(2);
    EmbeddingNode node(nh);
    spinner.start();
    ros::waitForShutdown();
    return 0;    
}