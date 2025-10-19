cmake_minimum_required(VERSION 3.15)
project(ECR VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenCV
find_package(OpenCV 4 REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Fetch CLI11
include(FetchContent)
FetchContent_Declare(
  cli11_proj
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(cli11_proj)


# Add calibration module
add_subdirectory(calibration)
