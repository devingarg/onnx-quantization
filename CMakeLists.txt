cmake_minimum_required(VERSION 3.13)

project(ONNX_Runtime_Inference VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(OpenCV REQUIRED)

# find_path() locates the full directory path to a specific file within the system
# the first argument is the variable name in which the result (full path) is stored
# the second argument is the name of the file we are looking for
# HINTS argument is used to prioritize looking into certain directories before searching in the default paths
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h)

# find_library() locates the full path to the library that needs to be linked to the project
# the actual filename starts with lib<name> and has the ".so" extension.
# E.g. if we're looking for libonnxruntime.so, we'd write onnxruntime in this find command
# This is useful because for other OSes it might be called <name>.dll or <name>.dylib so
# writing just the name here makes it OS-independent.
find_library(ONNX_RUNTIME_LIB onnxruntime)

# Name the executable that is to be created at the end of the build process
# the second argument is the file that is to be built into that executable
add_executable(classifier resnet_inference.cpp)

# include the headers for the libraries being used in the executable to be generated
target_include_directories(classifier PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

# include the libraries that need to be linked when the executable is run
target_link_libraries(classifier PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBS})