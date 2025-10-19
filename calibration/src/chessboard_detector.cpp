#include "calibration/chessboard_detector.h"

namespace ecr {

ChessboardDetector::ChessboardDetector(cv::Size board_size,
                                       float square_size_mm)
    : board_size_(board_size), square_size_mm_(square_size_mm) {}

ChessboardDetection ChessboardDetector::detect(const cv::Mat &image,
                                               bool refine) {
    ChessboardDetection result;
    result.success = false;

    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // Detect chessboard corners
    int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
    result.success =
        cv::findChessboardCorners(gray, board_size_, result.corners, flags);

    if (result.success && refine) {
        // Sub-pixel refinement
        cv::cornerSubPix(
            gray, result.corners, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30,
                             0.1));
    }

    result.visualization = image.clone();
    cv::drawChessboardCorners(result.visualization, board_size_, result.corners,
                              result.success);

    return result;
}

std::vector<cv::Point3f> ChessboardDetector::getObjectPoints() const {
    std::vector<cv::Point3f> points;
    points.reserve(board_size_.width * board_size_.height);

    for (int i = 0; i < board_size_.height; i++) {
        for (int j = 0; j < board_size_.width; j++) {
            points.emplace_back(j * square_size_mm_, i * square_size_mm_, 0.0f);
        }
    }

    return points;
}

} // namespace ecr
