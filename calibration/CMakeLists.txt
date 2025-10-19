cmake_minimum_required(VERSION 3.15)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Calibration library
add_library(ecr_calibration
    src/chessboard_detector.cpp
    src/camera_calibrator.cpp
    src/calibration_io.cpp
)

target_include_directories(ecr_calibration PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ecr_calibration PUBLIC
    ${OpenCV_LIBS}
)

# Calibration tool
add_executable(calibrate_camera
    tools/calibrate_camera.cpp
)

target_link_libraries(calibrate_camera PRIVATE
    ecr_calibration
    ${OpenCV_LIBS}
    CLI11::CLI11
)

# Verification tool
add_executable(verify_calibration
    tools/verify_calibration.cpp
)

target_link_libraries(verify_calibration PRIVATE
    ecr_calibration
    ${OpenCV_LIBS}
    CLI11::CLI11
)

install(TARGETS calibrate_camera verify_calibration DESTINATION bin)
