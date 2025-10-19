#pragma once
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>

namespace ecr {

std::vector<std::filesystem::path> findImages(const std::string &directory);

std::vector<cv::Mat> loadImages(const std::vector<std::filesystem::path> &paths);

} // namespace ecr
