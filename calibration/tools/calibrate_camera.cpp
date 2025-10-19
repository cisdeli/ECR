#include "calibration/calibration_io.h"
#include "calibration/camera_calibrator.h"
#include <CLI/CLI.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

int main(int argc, char **argv) {
    CLI::App app{"Camera calibration tool"};

    std::string input_dir;
    std::string output_yaml;
    int board_width;
    int board_height;
    float square_size;
    bool visualize = false;

    app.add_option("input_dir", input_dir, "Directory containing calibration images")
        ->required();
    app.add_option("output_yaml", output_yaml, "Output calibration file")
        ->required();
    app.add_option("board_width", board_width, "Chessboard internal corners width")
        ->required();
    app.add_option("board_height", board_height,
                   "Chessboard internal corners height")
        ->required();
    app.add_option("square_size", square_size, "Square size in mm")->required();
    app.add_flag("-v,--visualize", visualize, "Show detection visualization");

    CLI11_PARSE(app, argc, argv);

    ecr::CameraCalibrator calibrator(cv::Size(board_width, board_height),
                                     square_size);

    auto image_paths = ecr::findImages(input_dir);
    std::cout << "Found " << image_paths.size() << " images in " << input_dir
              << std::endl;

    for (const auto &path : image_paths) {
        std::cout << "\nProcessing: " << path.filename() << std::endl;
        cv::Mat image = cv::imread(path.string());

        if (image.empty()) {
            std::cerr << "Failed to load image: " << path << std::endl;
            continue;
        }

        calibrator.addImage(image, visualize);
    }

    if (calibrator.getNumImages() < 3) {
        std::cerr << "Error: Need at least 3 successful detections, got "
                  << calibrator.getNumImages() << std::endl;
        return 1;
    }

    auto intrinsics = calibrator.calibrate();
    intrinsics.save(output_yaml);

    std::cout << "\n=== Calibration Summary ===" << std::endl;
    std::cout << "Images used: " << intrinsics.num_images << std::endl;
    std::cout << "Image size: " << intrinsics.image_size << std::endl;
    std::cout << "Reprojection error: " << intrinsics.reprojection_error
              << " pixels" << std::endl;
    std::cout << "Output saved to: " << output_yaml << std::endl;

    cv::destroyAllWindows();
    return 0;
}
