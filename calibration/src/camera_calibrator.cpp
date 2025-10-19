#include "calibration/camera_calibrator.h"
#include <fstream>
#include <iostream>
#include <opencv2/calib3d.hpp>

namespace ecr {

CameraCalibrator::CameraCalibrator(cv::Size board_size, float square_size_mm)
    : detector_(board_size, square_size_mm), image_size_(0, 0) {}

bool CameraCalibrator::addImage(const cv::Mat &image, bool visualize) {
    auto detection = detector_.detect(image);

    if (!detection.success) {
        std::cerr << "Failed to detect chessboard in image" << std::endl;
        return false;
    }

    // Image size from first successful detection
    if (image_size_.width == 0) {
        image_size_ = image.size();
    } else if (image.size() != image_size_) {
        std::cerr << "Image size mismatch: expected " << image_size_ << ", got "
                  << image.size() << std::endl;
        return false;
    }

    // Store points
    image_points_.push_back(detection.corners);
    object_points_.push_back(detector_.getObjectPoints());

    std::cout << "Added image " << image_points_.size() << " ("
              << detection.corners.size() << " corners detected)" << std::endl;

    if (visualize) {
        cv::namedWindow("Chessboard Detection", cv::WINDOW_AUTOSIZE);
        cv::setWindowProperty("Chessboard Detection", cv::WND_PROP_ASPECT_RATIO,
                              cv::WINDOW_KEEPRATIO);
        cv::resizeWindow("Chessboard Detection", 1400,
                         900); // target size; Qt will scale content
        cv::imshow("Chessboard Detection", detection.visualization);
        cv::waitKey(500);
    }

    return true;
}

CameraIntrinsics CameraCalibrator::calibrate() {
    if (image_points_.size() < 3) {
        throw std::runtime_error("Need at least 3 images for calibration, got " +
                                 std::to_string(image_points_.size()));
    }

    CameraIntrinsics intrinsics;
    intrinsics.image_size = image_size_;
    intrinsics.num_images = image_points_.size();

    std::vector<cv::Mat> rvecs, tvecs;

    std::cout << "Running calibration with " << image_points_.size()
              << " images..." << std::endl;

    intrinsics.reprojection_error = cv::calibrateCamera(
        object_points_, image_points_, image_size_, intrinsics.camera_matrix,
        intrinsics.dist_coeffs, rvecs, tvecs,
        cv::CALIB_FIX_ASPECT_RATIO // Assume square pixels
    );

    std::cout << "Calibration complete!" << std::endl;
    std::cout << "Reprojection error: " << intrinsics.reprojection_error
              << " pixels" << std::endl;
    std::cout << "Camera matrix:\n"
              << intrinsics.camera_matrix << std::endl;
    std::cout << "Distortion coeffs: " << intrinsics.dist_coeffs.t() << std::endl;

    return intrinsics;
}

void CameraCalibrator::reset() {
    image_points_.clear();
    object_points_.clear();
    image_size_ = cv::Size(0, 0);
}

void CameraIntrinsics::save(const std::string &filepath) const {
    cv::FileStorage fs(filepath, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    fs << "image_width" << image_size.width;
    fs << "image_height" << image_size.height;
    fs << "camera_matrix" << camera_matrix;
    fs << "distortion_coeffs" << dist_coeffs;
    fs << "reprojection_error" << reprojection_error;
    fs << "num_images" << num_images;

    fs.release();
    std::cout << "Saved calibration to: " << filepath << std::endl;
}

CameraIntrinsics CameraIntrinsics::load(const std::string &filepath) {
    cv::FileStorage fs(filepath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    CameraIntrinsics intrinsics;

    int width, height;
    fs["image_width"] >> width;
    fs["image_height"] >> height;
    intrinsics.image_size = cv::Size(width, height);

    fs["camera_matrix"] >> intrinsics.camera_matrix;
    fs["distortion_coeffs"] >> intrinsics.dist_coeffs;
    fs["reprojection_error"] >> intrinsics.reprojection_error;
    fs["num_images"] >> intrinsics.num_images;

    fs.release();

    std::cout << "Loaded calibration from: " << filepath << std::endl;
    return intrinsics;
}

} // namespace ecr
