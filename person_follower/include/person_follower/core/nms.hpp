#ifndef FOLLOWER_CORE_NMS_HPP_
#define FOLLOWER_CORE_NMS_HPP_

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace person_follower {
    namespace core {
        /// @brief Apply NMS on float-precision boxes
        /// Taken from YOLOs-CPP (NMSBoxesF at core/nms.hpp) and modified to 
        /// fit my naming convention
        /// See: https://github.com/Geekgineer/YOLOs-CPP/blob/main/include/yolos/core/nms.hpp
        static inline void nms_boxes(const std::vector<cv::Rect2f>& boxes,
                                     const std::vector<float>& scores,
                                     float score_threshold,
                                     float nms_threshold,
                                     std::vector<int>& indices) {
            indices.clear();

            const size_t num_boxes = boxes.size();
            if (num_boxes == 0) {
                return;
            }

            // Step 1: Filter boxes by score threshold and create sorted indices
            std::vector<int> sorted_indices;
            sorted_indices.reserve(num_boxes);
            for (size_t i = 0; i < num_boxes; ++i) {
                if (scores[i] >= score_threshold) {
                    sorted_indices.push_back(static_cast<int>(i));
                }
            }

            if (sorted_indices.empty()) {
                return;
            }

            // Sort by score in descending order
            std::sort(sorted_indices.begin(), sorted_indices.end(),
                    [&scores](int a, int b) { return scores[a] > scores[b]; });

            // Step 2: Precompute areas
            std::vector<float> areas(num_boxes, 0.0f);
            for (size_t i = 0; i < num_boxes; ++i) {
                areas[i] = boxes[i].width * boxes[i].height;
            }

            // Step 3: Suppression
            std::vector<bool> suppressed(num_boxes, false);

            for (size_t i = 0; i < sorted_indices.size(); ++i) {
                int current_idx = sorted_indices[i];
                if (suppressed[current_idx]) {
                    continue;
                }

                indices.push_back(current_idx);

                const cv::Rect2f& current_box = boxes[current_idx];
                const float x1_max = current_box.x;
                const float y1_max = current_box.y;
                const float x2_max = current_box.x + current_box.width;
                const float y2_max = current_box.y + current_box.height;
                const float area_current = areas[current_idx];

                for (size_t j = i + 1; j < sorted_indices.size(); ++j) {
                    int compare_idx = sorted_indices[j];
                    if (suppressed[compare_idx]) {
                        continue;
                    }

                    const cv::Rect2f& compare_box = boxes[compare_idx];
                    const float x1 = std::max(x1_max, compare_box.x);
                    const float y1 = std::max(y1_max, compare_box.y);
                    const float x2 = std::min(x2_max, compare_box.x + compare_box.width);
                    const float y2 = std::min(y2_max, compare_box.y + compare_box.height);

                    const float inter_width = x2 - x1;
                    const float inter_height = y2 - y1;

                    if (inter_width <= 0 || inter_height <= 0) {
                        continue;
                    }

                    const float intersection = inter_width * inter_height;
                    const float union_area = area_current + areas[compare_idx] - intersection;
                    const float iou = (union_area > 0.0f) ? (intersection / union_area) : 0.0f;

                    if (iou > nms_threshold) {
                        suppressed[compare_idx] = true;
                    }
                }
            }
        }
    }
}

#endif // FOLLOWER_CORE_NMS_HPP_