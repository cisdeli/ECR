#pragma once

#include "calibration/chessboard_detector.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace ecr {

struct CameraIntrinsics {
    cv::Mat camera_matrix; // K matrix
    cv::Mat dist_coeffs;   // Distortion coefficients (k1,k2,p1,p2,k3)
    cv::Size image_size;
    double reprojection_error;
    int num_images;

    // For YAML serialization
    void save(const std::string &filepath) const;
    static CameraIntrinsics load(const std::string &filepath);
};

class CameraCalibrator {
  public:
    CameraCalibrator(cv::Size board_size, float square_size_mm);
    bool addImage(const cv::Mat &image, bool visualize = false);

    // Number of successfully added images
    int getNumImages() const { return image_points_.size(); }

    // Run calibration, at least 3 images required
    CameraIntrinsics calibrate();

    // Clear all collected data
    void reset();

  private:
    ChessboardDetector detector_;
    std::vector<std::vector<cv::Point2f>> image_points_;
    std::vector<std::vector<cv::Point3f>> object_points_;
    cv::Size image_size_;
};

} // namespace ecr
