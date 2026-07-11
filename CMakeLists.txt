cmake_minimum_required(VERSION 3.16)
project(person_follower)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)



###-============-###
### Dependencies ###
###-============-###
include(${CMAKE_CURRENT_LIST_DIR}/person_follower/person_follower.cmake)

find_package(
    catkin REQUIRED COMPONENTS
        
        person_follower_msgs

        roscpp
        roslib
        cv_bridge
        std_msgs
        sensor_msgs
        geometry_msgs
        visualization_msgs
        message_filters
        dynamic_reconfigure
)

generate_dynamic_reconfigure_options(
    cfg/Pose.cfg
    cfg/Tracking.cfg
)


catkin_package(
    CATKIN_DEPENDS
        person_follower_msgs

        roscpp
        roslib
        cv_bridge
        std_msgs
        sensor_msgs
        geometry_msgs
        visualization_msgs
        message_filters
)




###-================-###
### Helper functions ###
###-================-###
function(add_node name file)
    add_executable(${name} ${file})

    add_dependencies(${name}
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
    )

    target_link_libraries(${name} PRIVATE
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
    )

    target_include_directories(${name} PRIVATE
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    # Following function is from person_follower/person_follower.cmake
    # The same .cmake file which handles the ONNX Runtime dependency
    target_link_person_follower(${name})
endfunction()



function(add_py_node file)
    catkin_install_python(PROGRAMS ${file}
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
endfunction()



###-=====-###
### Nodes ###
###-=====-###
add_node(body_pose src/body_pose.cpp)
add_node(simple_full_pose src/naive/simple_full_pose.cpp)

# add_py_node(scripts/hands.py)
