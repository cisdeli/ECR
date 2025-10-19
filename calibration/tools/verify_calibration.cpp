#include "calibration/calibration_io.h"
#include "calibration/camera_calibrator.h"
#include "calibration/chessboard_detector.h"
#include <CLI/CLI.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

struct AnnotatedPoint {
    cv::Point2f pt;
    std::string text;
};

struct MouseData {
    cv::Mat baseImage;
    std::vector<AnnotatedPoint> annotations;
};

cv::Point2f project3DPoint(const cv::Point3f &objPnt, const cv::Mat &rvec,
                           const cv::Mat &tvec, const cv::Mat &K,
                           const cv::Mat &D) {
    std::vector<cv::Point3f> objPnts = {objPnt};
    std::vector<cv::Point2f> imgPnts;
    cv::projectPoints(objPnts, rvec, tvec, K, D, imgPnts);
    return imgPnts[0];
}

void onMouse(int event, int x, int y, int flags, void *userdata) {
    MouseData *data = static_cast<MouseData *>(userdata);

    cv::Mat display = data->baseImage.clone();

    if (event == cv::EVENT_MOUSEMOVE) {
        double minDist = std::numeric_limits<double>::max();
        int closestIdx = -1;

        for (size_t i = 0; i < data->annotations.size(); ++i) {
            double dx = x - data->annotations[i].pt.x;
            double dy = y - data->annotations[i].pt.y;
            double dist = std::sqrt(dx * dx + dy * dy);
            if (dist < minDist) {
                minDist = dist;
                closestIdx = i;
            }
        }

        if (closestIdx >= 0 && minDist < 20) {
            const auto &ann = data->annotations[closestIdx];
            cv::Point textPos = ann.pt + cv::Point2f(10, -10);

            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.5;
            int thickness = 1;
            int baseline = 0;
            cv::Size textSize =
                cv::getTextSize(ann.text, fontFace, fontScale, thickness, &baseline);

            cv::rectangle(display, textPos + cv::Point(0, baseline),
                          textPos + cv::Point(textSize.width, -textSize.height),
                          cv::Scalar(255, 255, 255), cv::FILLED);
            cv::putText(display, ann.text, textPos, fontFace, fontScale,
                        cv::Scalar(0, 0, 0), thickness);
        }
    }

    cv::imshow("Reprojection", display);
}

void plotPoints(const std::vector<std::vector<cv::Point2f>> &imgPntsVec,
                const std::vector<std::vector<cv::Point3f>> &objPntsVec,
                const cv::Mat &K, const cv::Mat &D,
                const std::vector<cv::Mat> &rvec,
                const std::vector<cv::Mat> &tvec,
                const std::vector<cv::Mat> &images) {

  for (size_t view = 0; view < images.size(); ++view) {
    cv::Mat imgCopy = images[view].clone();
    if (imgCopy.channels() == 1)
      cv::cvtColor(imgCopy, imgCopy, cv::COLOR_GRAY2BGR);

    double knownRealDistance_mm = 1.0;
    double distx = imgPntsVec[view][0].x - imgPntsVec[view][1].x;
    double disty = imgPntsVec[view][0].y - imgPntsVec[view][1].y;
    double pixelDist = std::sqrt(distx * distx + disty * disty);
    double alpha = knownRealDistance_mm / pixelDist;

    std::vector<AnnotatedPoint> annotations;
    for (size_t j = 0; j < imgPntsVec[view].size(); ++j) {
      cv::Point2f observed = imgPntsVec[view][j];
      cv::Point2f proj =
          project3DPoint(objPntsVec[view][j], rvec[view], tvec[view], K, D);
      double dx = proj.x - observed.x;
      double dy = proj.y - observed.y;
      double errorPixels = std::sqrt(dx * dx + dy * dy);
      double errorWU = errorPixels * alpha;

      std::string errorStr =
          cv::format("Error: %.2f px, %.2f mm", errorPixels, errorWU);

      cv::circle(imgCopy, observed, 5, cv::Scalar(0, 0, 255), -1);
      cv::circle(imgCopy, proj, 5, cv::Scalar(0, 255, 0), -1);
      cv::line(imgCopy, observed, proj, cv::Scalar(0, 0, 0), 1);

      AnnotatedPoint ann;
      ann.pt = proj;
      ann.text = errorStr;
      annotations.push_back(ann);
    }

    MouseData mouseData;
    mouseData.baseImage = imgCopy;
    mouseData.annotations = annotations;

    cv::Point textPosition(10, 30);

    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.0;
    int thickness = 2;
    int baseline = 0;
    std::string instructionsStr =
        "Red Points: Ground Truth; Green Points: Reprojected Points. Hover any "
        "point to display it's error. Press any key to display the next image.";
    cv::Size textSize = cv::getTextSize(instructionsStr, fontFace, fontScale,
                                        thickness, &baseline);
    cv::rectangle(imgCopy, textPosition + cv::Point(0, baseline),
                  textPosition + cv::Point(textSize.width, -textSize.height),
                  cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(imgCopy, instructionsStr, textPosition, fontFace, fontScale,
                cv::Scalar(0, 0, 0), thickness);

    cv::namedWindow("Reprojection", cv::WINDOW_NORMAL);
    cv::resizeWindow("Reprojection", imgCopy.cols / 4, imgCopy.rows / 4);
    cv::setMouseCallback("Reprojection", onMouse, &mouseData);
    cv::imshow("Reprojection", imgCopy);
    cv::waitKey(0);
    cv::destroyWindow("Reprojection");
  }
}

int main(int argc, char **argv) {
  CLI::App app{"Calibration verification tool"};

  std::string input_dir;
  std::string calib_yaml;
  int board_width;
  int board_height;
  float square_size;

  app.add_option("input_dir", input_dir,
                 "Directory containing verification images")
      ->required();
  app.add_option("calibration_yaml", calib_yaml, "Calibration file to verify")
      ->required();
  app.add_option("board_width", board_width, "Chessboard internal corners width")
      ->required();
  app.add_option("board_height", board_height,
                 "Chessboard internal corners height")
      ->required();
  app.add_option("square_size", square_size, "Square size in mm")->required();

  CLI11_PARSE(app, argc, argv);

  ecr::CameraIntrinsics intrinsics = ecr::CameraIntrinsics::load(calib_yaml);
  ecr::ChessboardDetector detector(cv::Size(board_width, board_height),
                                   square_size);

  auto image_paths = ecr::findImages(input_dir);
  std::cout << "Found " << image_paths.size() << " images in " << input_dir
            << std::endl;

  std::vector<std::vector<cv::Point2f>> image_points;
  std::vector<std::vector<cv::Point3f>> object_points;
  std::vector<cv::Mat> images;

  for (const auto &path : image_paths) {
    std::cout << "Processing: " << path.filename() << std::endl;
    cv::Mat image = cv::imread(path.string());

    if (image.empty()) {
      std::cerr << "Failed to load: " << path << std::endl;
      continue;
    }

    auto detection = detector.detect(image);
    if (!detection.success) {
      std::cerr << "No chessboard detected in: " << path.filename()
                << std::endl;
      continue;
    }

    image_points.push_back(detection.corners);
    object_points.push_back(detector.getObjectPoints());
    images.push_back(image);

    std::cout << "Detected " << detection.corners.size() << " corners"
              << std::endl;
  }

  if (images.empty()) {
    std::cerr << "No valid detections found" << std::endl;
    return 1;
  }

  std::vector<cv::Mat> rvecs, tvecs;
  for (size_t i = 0; i < images.size(); ++i) {
    cv::Mat rvec, tvec;
    cv::solvePnP(object_points[i], image_points[i], intrinsics.camera_matrix,
                 intrinsics.dist_coeffs, rvec, tvec);
    rvecs.push_back(rvec);
    tvecs.push_back(tvec);
  }

  std::cout << "\nComputing reprojection errors..." << std::endl;

  double totalError = 0.0;
  int totalPoints = 0;

  for (size_t i = 0; i < images.size(); ++i) {
    std::vector<cv::Point2f> reprojected;
    cv::projectPoints(object_points[i], rvecs[i], tvecs[i],
                      intrinsics.camera_matrix, intrinsics.dist_coeffs,
                      reprojected);

    double imgError = 0.0;
    for (size_t j = 0; j < image_points[i].size(); ++j) {
      double dx = reprojected[j].x - image_points[i][j].x;
      double dy = reprojected[j].y - image_points[i][j].y;
      double error = std::sqrt(dx * dx + dy * dy);
      imgError += error;
      totalError += error;
      totalPoints++;
    }

    std::cout << "Image " << i << ": " << imgError / image_points[i].size()
              << " px" << std::endl;
  }

  std::cout << "\n=== Verification Summary ===" << std::endl;
  std::cout << "Images processed: " << images.size() << std::endl;
  std::cout << "Total points: " << totalPoints << std::endl;
  std::cout << "Mean reprojection error: " << totalError / totalPoints << " px"
            << std::endl;
  std::cout << "Calibration reprojection error: "
            << intrinsics.reprojection_error << " px" << std::endl;

  plotPoints(image_points, object_points, intrinsics.camera_matrix,
             intrinsics.dist_coeffs, rvecs, tvecs, images);

  return 0;
}
