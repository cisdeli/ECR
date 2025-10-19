#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace ecr {

struct ChessboardDetection {
    std::vector<cv::Point2f> corners;
    cv::Mat visualization;
    bool success;
};

class ChessboardDetector {
  public:
    ChessboardDetector(cv::Size board_size, float square_size_mm);

    // Detect chessboard in image
    ChessboardDetection detect(const cv::Mat &image, bool refine = true);

    // Generate 3D object points for this board
    std::vector<cv::Point3f> getObjectPoints() const;

    // Getters
    cv::Size getBoardSize() const { return board_size_; }
    float getSquareSize() const { return square_size_mm_; }

  private:
    cv::Size board_size_;  // Internal corners (width, height)
    float square_size_mm_; // Square size in millimeters
};

} // namespace ecr
