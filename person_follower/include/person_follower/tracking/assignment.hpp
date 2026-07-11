#ifndef PERSON_FOLLOWER_TRACKING_ASSIGMNET_HPP_
#define PERSON_FOLLOWER_TRACKING_ASSIGMNET_HPP_

#include <opencv2/core.hpp>

#include <cstdint>
#include <limits>
#include <vector>

namespace person_follower {
    namespace tracking {
        /// @brief Compute IoU on 2 float-precision boxes
        static inline float iou_box(const cv::Rect2f& a, const cv::Rect2f& b) {
            const float inter_a = (a & b).area();
            const float union_a = a.area() + b.area() - inter_a;
            return (union_a > 0.0f) ? inter_a / union_a : 0.0f;
        }

        /// @brief Kuhn-Munkeres optimal min-cost assignment in O(n^3)
        std::vector<size_t> hungarian(const cv::Mat_<float>& costs) {
            constexpr float  INF = std::numeric_limits<float>::max();
            constexpr size_t MAX = std::numeric_limits<size_t>::max();

            const size_t rows = static_cast<size_t>(costs.rows);
            const size_t cols = static_cast<size_t>(costs.cols);
            const size_t n = std::max(rows, cols);

            if ( rows == 0 || cols == 0 )
                return std::vector<size_t>(rows, SIZE_MAX);

            // Square 1-indexed cost matrix
            cv::Mat_<float> a;
            if ( rows == cols ) {
                a = costs;
            }
            else {
                double min_cost, max_cost;
                cv::minMaxLoc(costs, &min_cost, &max_cost);
                const float pad = static_cast<float>(max_cost) + 1.0f;
                a = cv::Mat_<float>(n, n, pad);
                cv::Mat_<float> roi = a(cv::Rect(0, 0, cols, rows));
                costs.copyTo(roi);
            }

            // Where index is 0 := "unmatched"
            std::vector<float>  u(n + 1, 0.0f); //< Row potentials
            std::vector<float>  v(n + 1, 0.0f); //< Column potentials
            std::vector<size_t> p(n + 1, 0);    //< p[j] Row matched to column j
            std::vector<size_t> way(n + 1, 0);  //< way[j] Predecessor column

            for ( size_t i = 1; i <= n; i++ ) {
                p[0] = i;

                std::vector<float> minv(n + 1, INF);
                std::vector<bool>  used(n + 1, false);

                size_t j0 = 0;
                do {
                    used[j0] = true;

                    const size_t i0 = p[j0];
                    
                    float delta = INF;
                    size_t j1   = MAX;

                    for ( size_t j = 1; j <= n; j++ ) {
                        if ( used[j] )
                            continue;

                        const float cur = a(i0 - 1, j - 1) - u[i0] - v[j];
                        if ( cur < minv[j] ) {
                            minv[j] = cur;
                            way[j]  = j0;
                        }

                        if ( minv[j] < delta ) {
                            delta = minv[j];
                            j1    = j;
                        }
                    }

                    // Update potentials
                    for ( size_t j = 0; j <= n; j++ ) {
                        if ( used[j] ) {
                            u[p[j]] += delta;
                            v[j]    -= delta;
                        }
                        else {
                            minv[j] -= delta;
                        }
                    }

                    j0 = j1;
                } while ( p[j0] != 0 );

                // Augment: Walk predecessors back to the start
                do {
                    const size_t j1 = way[j0];
                    p[j0] = p[j1];
                    j0 = j1;
                } while ( j0 );
                }

            // p[j] = row matched to column j
            std::vector<size_t> assigned(rows, MAX); //< MAX := unassigned
            for ( size_t j = 1; j <= n; j++ ) {
                const size_t i = p[j];
                if ( i >= 1 && i <= rows && j <= cols )
                    assigned[i - 1] = j - 1;
            }

            return assigned;
        }

        /// @brief Apply hungraian algorithm, filtering out any costs that are too large
        std::vector<size_t> thresholded_hungarian(const cv::Mat_<float>& costs, 
                                      float threshold = std::numeric_limits<float>::max()) {
            std::vector<size_t> assignment = hungarian(costs);

            for ( size_t i = 0; i < static_cast<size_t>(costs.rows); i++ ) {
                size_t j = assignment[i];

                if ( j >= static_cast<size_t>(costs.cols) ) continue;
                float c = costs(i, j);
                if ( c > threshold ) assignment[i] = SIZE_MAX;
            }

            return assignment;
        }

        /// @brief Hungrian assignment of overlapping bounding boxes
        /// Position-aligned with `from` - output is same size as `from`
        /// MAX_SIZE means no good assignment 
        std::vector<size_t> hungarian(const std::vector<cv::Rect2f>& from, const std::vector<cv::Rect2f>& to, float iou_threshold = 0.0f) {
            cv::Mat_<float> costs(from.size(), to.size(), 0.0f);
            
            for ( size_t f = 0; f < from.size(); f++ )
                for ( size_t t = 0; t < to.size(); t++ )
                    costs(f, t) = 1.0f - iou_box(from[f], to[t]);

            return thresholded_hungarian(costs, 1.0f - iou_threshold);
        }

        /// @brief Hungarian assignment based on distances
        std::vector<size_t> hungarian(const std::vector<cv::Point2f>& from, const std::vector<cv::Point2f>& to, float dist_threshold = std::numeric_limits<float>::max()) {
            cv::Mat_<float> costs(from.size(), to.size(), 0.0f);

            for ( size_t f = 0; f < from.size(); f++ ) {
                for ( size_t t = 0; t < to.size(); t++ ) {
                    const float dx = from[f].x - to[t].x;
                    const float dy = from[f].y - to[t].y;
                    costs(f, t) = std::sqrt(dx*dx + dy*dy);
                }
            }

            return thresholded_hungarian(costs, dist_threshold);
        }
    }
}

#endif // FOLLOWER_TRACKING_ASSIGMNET_HPP_