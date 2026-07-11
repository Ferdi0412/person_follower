#ifndef PERSON_FOLLOWER_TRACKING_SORT_HPP_
#define PERSON_FOLLOWER_TRACKING_SORT_HPP_

#include "person_follower/tracking/assignment.hpp"

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include <cstdint>
#include <vector>

namespace person_follower {
    namespace tracking {
        /**
         * @brief Implements SORT algorithm for tracking bounding boxes
         */
        class SortTracker {
            public:
                /// @brief Info on results of last call to update
                struct UpdateInfo {
                    std::vector<uint32_t> ids;       //< Tracking ids for each input - 0 := untracked
                    std::vector<size_t>   added;     //< Index of newly tracked inputs
                    std::vector<size_t>   untracked; //< Index of yet untracked inputs
                    std::vector<uint32_t> dropped;   //< Tracking ids for expired boxes
                };

                /// @brief Match new detections with previous detections
                /// @param detections Boxes to match
                UpdateInfo update(const std::vector<cv::Rect2f>& detections, float iou_threshold = 0.3f, int min_hits = 3, int max_age = 3) {
                    UpdateInfo results;
                    results.ids = std::vector<uint32_t>(detections.size(), 0);

                    // Predict all positions
                    std::vector<cv::Rect2f> predictions(trackers.size());

                    for ( size_t i = 0; i < trackers.size(); i++ )
                        predictions[i] = trackers[i].filter.predict();

                    // // Detections (row) and predcitions (column) assignment
                    // cv::Mat_<float> costs(detections.size(), predictions.size(), 0.0f);

                    // for ( size_t d = 0; d < detections.size(); d++ ) 
                    //     for ( size_t p = 0; p < predictions.size(); p++ )
                    //         costs(d, p) = 1.0f - iou_box(detections[d], predictions[p]);

                    // std::vector<size_t> assigned = hungarian(costs);

                    // // Filter assignments
                    // for ( size_t d = 0; d < detections.size(); d++ ) {
                    //     size_t p = assigned[d];

                    //     if ( p >= predictions.size() ) //< Unassigned
                    //         continue;

                    //     float iou = 1.0f - costs(d, p);

                    //     // Mark those below iou threshold as "unassigned"
                    //     if ( iou < iou_threshold ) // 1 - costs < t := cost > 1 - t
                    //         assigned[d] = SIZE_MAX;
                    // }

                    std::vector<size_t> assigned = hungarian(detections, predictions, iou_threshold);

                    // Update trackers based on detections
                    std::vector<bool> trackers_updated(trackers.size(), false);
                    for ( size_t d = 0; d < detections.size(); d++ ) {
                        size_t t = assigned[d];

                        if ( t < trackers.size() /* Matched */ ) {
                            auto& tracker = trackers[t];

                            tracker.filter.update(detections[d]);

                            tracker.time_since_update = 0;
                            tracker.hit_streak ++;
                            tracker.hits ++;

                            // Added
                            if ( tracker.id == 0 && tracker.hits > min_hits ) {
                                tracker.id = next_id();
                                results.added.push_back(d);
                            }

                            results.ids[d] = tracker.id;

                            trackers_updated[t] = true;
                        }

                        else {
                            // Untracked
                            trackers.emplace_back(detections[d], filter_settings);
                            results.untracked.push_back(d);
                        }
                    }

                    // Update unmatched trackers
                    for ( size_t idx_plus = trackers_updated.size(); idx_plus > 0; idx_plus -- ) {
                        size_t t = idx_plus - 1;
                        if ( !trackers_updated[t] ) {
                            trackers[t].hit_streak = 0;
                            trackers[t].time_since_update ++;

                            // Dropped
                            if ( trackers[t].time_since_update > max_age ) {
                                if ( trackers[t].id )
                                    results.dropped.push_back(trackers[t].id);
                                trackers.erase(trackers.begin() + t);
                            }
                        }
                    }

                    return results;
                }

                /// @brief Compute IoU on 2 float-precision boxes
                static inline float iou_box(const cv::Rect2f& a, const cv::Rect2f& b) {
                    const float inter_a = (a & b).area();
                    const float union_a = a.area() + b.area() - inter_a;
                    return (union_a > 0.0f) ? inter_a / union_a : 0.0f;
                }



                /**
                 * =====================================
                 * === SortTracker::KalmanFilterSettings ===
                 * =====================================
                 * @brief Settings for tuning the kalman filter
                 */
                struct KalmanFilterSettings {
                    std::array<float, 7> q;
                    std::array<float, 4> r;
                    std::array<float, 7> p;

                    static const std::array<float, 7>& default_q() {
                        static const std::array<float, 7> q_ = {{
                            1e-2f, 1e-2f,       //< cx, cy 
                            1e-2f, 1e-2f,       //< s,  r
                            1e-1f, 1e-1f, 1e-2f //< vx, vy, vs
                        }};
                        return q_;
                    }

                    static const std::array<float, 4>& default_r() {
                        static const std::array<float, 4> r_ = {{
                            0.1f,  0.1f,  //< cx, cy
                            10.0f, 10.0f //< s, r   - trust less than position
                        }};
                        return r_;
                    }

                    static const std::array<float, 7>& default_p() {
                        static const std::array<float, 7> p_ = {{
                            1.0f,    1.0f,            //< cx, cy
                            1.0f,    1.0f,            //< s,  r
                            1000.0f, 1000.0f, 1000.0f //< vx, vy, vs - update quickly initially
                        }};
                        return p_;
                    }

                    KalmanFilterSettings()
                        : q(default_q()), r(default_r()), p(default_p()) {}

                    KalmanFilterSettings(
                        const std::array<float, 7>& q_, 
                        const std::array<float, 4>& r_
                    ) : q(q_), r(r_), p(default_p()) {} 

                    KalmanFilterSettings(
                        const std::array<float, 7>& q_,
                        const std::array<float, 4>& r_,
                        const std::array<float, 7>& p_
                    ) : q(q_), r(r_), p(p_) {}
                };



                /**
                 * =============================
                 * === SortTracker::KalmanFilter === 
                 * =============================
                 * @brief Kalman filter model for tracking a single box
                 * 
                 * The model assumes linear dynamics, with constant inte
                 * -rvals. The filter state consists of 7 values (only 
                 * the first 4 are used for updates):
                 *  - cx (centroid x-)
                 *  - cy (centroid y-)
                 *  - s  (size or area: width * height)
                 *  - r  (aspect ratio: width / height)
                 *  - vx (velocity of centroid x-)
                 *  - vy (velocity of centroid y-)
                 *  - vs (rate of change of size)
                 */
                class KalmanFilter {
                    public:
                        KalmanFilter(
                            const cv::Rect2f& initial_box,
                            const KalmanFilterSettings& setting = {}
                        ) : kf(7, 4, 0, CV_32F) {
                            // transitionMatrix "F" is 7x7
                            cv::setIdentity(kf.transitionMatrix);
                            kf.transitionMatrix.at<float>(0, 4) = 1.0f; // cx += vx
                            kf.transitionMatrix.at<float>(1, 5) = 1.0f; // cy += vy
                            kf.transitionMatrix.at<float>(2, 6) = 1.0f; // s += vs

                            // measurementMatrix "H" is 4x7
                            // kf.measurementMatrix.setTo(0);
                            kf.measurementMatrix = cv::Mat::zeros(4, 7, CV_32F);
                            for ( int i = 0; i < 4; i++ )
                                kf.measurementMatrix.at<float>(i, i) = 1.0f;

                            // processNoiseCov "Q" is 7x7
                            kf.processNoiseCov.setTo(0);
                            for ( int i = 0; i < 7; i++ )
                                kf.processNoiseCov.at<float>(i, i) = setting.q[i];

                            // measruementNoiseCov "R" is 4x4
                            cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1.0f));
                            for ( int i = 0; i < 4; i++ )
                                kf.measurementNoiseCov.at<float>(i, i) = setting.r[i];

                            // errorCovPost "P" 7x7
                            cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1.0f));
                            for ( int i = 0; i < 7; i++ )
                                kf.errorCovPost.at<float>(i, i) = setting.p[i];

                            // Set initial box state
                            float w = initial_box.width;
                            float h = initial_box.height;
                            kf.statePost.at<float>(0) = initial_box.x + w / 2.0f;
                            kf.statePost.at<float>(1) = initial_box.y + w / 2.0f;
                            kf.statePost.at<float>(2) = w * h;
                            kf.statePost.at<float>(3) = (h > 0.0f) ? w / h : 0.0f;
                            kf.statePost.at<float>(4) = 0.0f;
                            kf.statePost.at<float>(5) = 0.0f;
                            kf.statePost.at<float>(6) = 0.0f;
                        }

                        cv::Rect2f last() const {
                            return from_state(kf.statePost);
                        }

                        cv::Point2f position() const {
                            return cv::Point2f(
                                kf.statePost.at<float>(0), //< x
                                kf.statePost.at<float>(1)  //< y
                            );
                        }

                        cv::Vec2f velocity() const {
                            return cv::Vec2f(
                                kf.statePost.at<float>(4), //< vx
                                kf.statePost.at<float>(5)  //< vy
                            );
                        }

                        cv::Rect2f predict() {
                            return from_state(kf.predict());
                        }

                        void update(const cv::Rect2f& box) {
                            float cx = box.x + box.width / 2.0f;
                            float cy = box.y + box.height / 2.0f;
                            float s  = box.width * box.height;
                            float r  = (box.height > 0.0f) ? box.width / box.height : 0.0f;
                            
                            cv::Mat measured = (cv::Mat_<float>(4, 1) << cx, cy, s, r);
                            kf.correct(measured);

                            // Guard against negative area
                            if ( kf.statePre.at<float>(2) + kf.statePost.at<float>(6) <= 0.0f )
                                kf.statePost.at<float>(6) = 0.0f;
                        }

                    private:    
                        inline cv::Rect2f from_state(const cv::Mat& state) const {
                            float s = state.at<float>(2);
                            float r = state.at<float>(3);
                            float w = std::sqrt(s * r);
                            float h = (w > 0.0f) ? s / w : 0.0f;

                            float x = state.at<float>(0) - w / 2.0f;
                            float y = state.at<float>(1) - h / 2.0f;

                            return cv::Rect2f(x, y, w, h);
                        }

                    private:
                        cv::KalmanFilter kf;
                };



                /**
                 * ========================
                 * === SortTracker::Tracked ===
                 * ========================
                 * @brief Everything used for tracking
                 */
                /// @brief A tracked instance, with lifetime args for tracking
                struct Tracked {
                    uint32_t id = 0;
                    KalmanFilter filter;

                    uint32_t hits = 0;
                    uint32_t hit_streak = 0;
                    
                    uint32_t time_since_update = 0;

                    /// @brief Constructor needs initial box and settings
                    Tracked(const cv::Rect2f& box, const KalmanFilterSettings& settings) 
                        : filter(box, settings) {}

                    /// @brief Get bounding box
                    cv::Rect2f get_box() const {
                        return filter.last();
                    }

                    /// @brief Get centroid
                    cv::Point2f get_center() const {
                        return filter.position();
                    }

                    /// @brief Get velocity
                    cv::Vec2f get_velocity() const {
                        return filter.velocity();
                    }
                    
                    /// @brief Draw current bounding box
                    void draw(cv::Mat& image, bool show_id = false) const {
                        cv::Rect box = get_box();
                        cv::Scalar color = id ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 0);
                        cv::rectangle(image, box, color);
                        if ( show_id ) {
                            cv::putText(
                                image,
                                cv::format("%d", id),
                                box.tl(),
                                cv::FONT_HERSHEY_DUPLEX,
                                0.5,
                                cv::Scalar(0, 0, 255)
                            );
                        }
                    }
                };



                /* === SortTracker Getters === */
                /// @brief Expose the underlying data 
                const std::vector<Tracked>& data() const {
                    return trackers;
                }

                /// @brief Draw all boxes onto an image
                const void draw(cv::Mat& image, bool show_id = false) const {
                    for ( const auto& t: trackers )
                        t.draw(image, show_id);
                }

                /// @brief Get the count/size
                size_t size() const noexcept {
                    return trackers.size();
                }

                /// @brief Get the count/size
                size_t count() const noexcept {
                    return trackers.size();
                }

            private:
                /// @brief Should be fairly safe - expect objects not to last until wraparound 
                uint32_t next_id() {
                    if ( ++id_counter == 0 ) 
                        ++id_counter; //< Skip 0
                    return id_counter;
                }

                // === Dynamic Variables ===
                uint32_t id_counter = 0;
                std::vector<Tracked> trackers;

                // === Settings === 
                KalmanFilterSettings filter_settings;

            public:
                /** === Constructor === ============================================== */
                SortTracker(const KalmanFilterSettings& kfs = {})
                    : filter_settings(kfs) {}
        }; // ~SortTracker
    }
}

#endif // FOLLOWER_TRACKING_SORT_HPP_