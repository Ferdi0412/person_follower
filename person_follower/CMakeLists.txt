cmake_minimum_required(VERSION 3.16)
project(follower)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_CURRENT_LIST_DIR}/follower.cmake)



###-================-###
### Helper functions ###
###-================-###
function(add_follower_executable name file)
    add_executable(${name} ${file})

    # Following function is from follower.cmake
    target_link_follower(${name})
endfunction()



###-========-###
### Programs ###
###-========-###
### Benchmarking
add_follower_executable(yolo_benchmark bencmarking/yolo.cpp)