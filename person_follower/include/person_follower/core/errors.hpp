#ifndef FOLLOWER_CORE_ERRORS_HPP_
#define FOLLOWER_CORE_ERRORS_HPP_

#include <stdexcept>

namespace person_follower {
    /// @brief Thrown by various classes
    /// Only non-inherited exception thrown will be std::argument_error
    class FollowerException: public std::runtime_error {
        public:
            explicit FollowerException(const std::string& error)
                : std::runtime_error(error) {}
    };

    /// @brief ONNX model loaded is invalid or unsupported
    class InvalidModel: public FollowerException {
        public:
            explicit InvalidModel(const std::string& cls, const std::string& file, const std::string& msg)
                : FollowerException("InvalidModel <" + cls + " '" + file + "'>: " + msg) {}
    };

    /// @brief Thrown by classes when more than an image is passed
    class AlignmentFailed: public FollowerException {
        public:
            explicit AlignmentFailed(const std::string& error)
                : FollowerException("AlignmentFailed: " + error) {}
    };
}

#endif // FOLLOWER_CORE_ERRORS_HPP_