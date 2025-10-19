#include "calibration/calibration_io.h"
#include <algorithm>
#include <iostream>

namespace ecr {

std::vector<std::filesystem::path> findImages(const std::string &directory) {
    namespace fs = std::filesystem;
    std::vector<fs::path> image_paths;

    for (const auto &entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                image_paths.push_back(entry.path());
            }
        }
    }

    std::sort(image_paths.begin(), image_paths.end());
    return image_paths;
}

std::vector<cv::Mat> loadImages(const std::vector<std::filesystem::path> &paths) {
    std::vector<cv::Mat> images;
    images.reserve(paths.size());

    for (const auto &path : paths) {
        cv::Mat img = cv::imread(path.string());
        if (!img.empty()) {
            images.push_back(img);
        } else {
            std::cerr << "Warning: Failed to load " << path << std::endl;
        }
    }

    return images;
}

} // namespace ecr
