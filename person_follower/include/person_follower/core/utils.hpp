#ifndef FOLLOWER_CORE_UTILS_HPP_
#define FOLLOWER_CORE_UTILS_HPP_

#include <string>
#include <chrono>

namespace person_follower {
    namespace utils {
        /// @brief Remove last 'n' path components
        inline std::string parent_dir(std::string path, size_t n = 1) {
            for ( size_t i = 0; i < n; i++ ) {
                size_t slash = path.rfind('/');
                if ( slash == std::string::npos )
                    return "";
                path.resize(slash);
            }
            return path;
        }

        /// @brief Prepend with the path to the follower library directory
        std::string person_follower_dir(const std::string& path) {
            #ifdef PERSON_FOLLOWER_DIR
            static const std::string base = PERSON_FOLLOWER_DIR;
            #else
            #warning "PERSON_FOLLOWER_DIR is not set"
            // __FILE__ := "{PERSON_FOLLOWER_DIR}/include/follower/utils/directory.hpp"
            static const std::string base = parent_dir(__FILE__, 4);
            #endif

            if ( path.empty() ) return base;
            if ( path.front() == '/' ) return base + path;
            return base + "/" + path;
        }
    }
}

#endif // FOLLOWER_CORE_UTILS_HPP_