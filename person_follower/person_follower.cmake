### === ONNX RUNTIME === ===============================================
# This was taken from the great YOLOs-CPP project

set(ONNXRUNTIME_SEARCH_PATHS
    "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.20.1"
    "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-gpu-1.20.1"
    "${CMAKE_SOURCE_DIR}/onnxruntime"
    "/opt/onnxruntime"
    "$ENV{ONNXRUNTIME_DIR}"
)
    
foreach(path ${ONNXRUNTIME_SEARCH_PATHS})
    if(EXISTS "${path}/include/onnxruntime_cxx_api.h")
        set(ONNXRUNTIME_DIR "${path}")
        break()
    endif()
endforeach()

# Convert relative path to absolute (relative paths are resolved from CMAKE_CURRENT_BINARY_DIR)
if(ONNXRUNTIME_DIR AND NOT IS_ABSOLUTE "${ONNXRUNTIME_DIR}")
    get_filename_component(ONNXRUNTIME_DIR "${ONNXRUNTIME_DIR}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
endif()

if(NOT ONNXRUNTIME_DIR OR NOT EXISTS "${ONNXRUNTIME_DIR}/include")
    message(FATAL_ERROR 
        "ONNX Runtime not found!\n"
        "Please set ONNXRUNTIME_DIR to your ONNX Runtime installation:\n"
        "  cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime\n"
        "Or download it from: https://github.com/microsoft/onnxruntime/releases"
    )
endif()


set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")


### === Library for ease-of-use === ====================================
find_package(OpenCV REQUIRED)


add_library(person_follower_lib INTERFACE)

target_include_directories(person_follower_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    $<INSTALL_INTERFACE:include>
    "${CMAKE_SOURCE_DIR}/include"
    "${ONNXRUNTIME_DIR}/include"
)

target_link_libraries(person_follower_lib INTERFACE 
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
)

target_include_directories(person_follower_lib INTERFACE
    ${OpenCV_INCLUDE_DIRS}
)

get_filename_component(PERSON_FOLLOWER_PARENT_DIR "${CMAKE_CURRENT_LIST_DIR}" DIRECTORY)

target_compile_definitions(person_follower_lib INTERFACE
    PERSON_FOLLOWER_DIR="${PERSON_FOLLOWER_PARENT_DIR}"
)

function(target_link_person_follower name)
    target_link_libraries(${name} PRIVATE person_follower_lib)

    set_target_properties(${name} PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
    )
endfunction()