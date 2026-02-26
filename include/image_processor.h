#pragma once

#include <opencv2/core.hpp>
#include <string>

struct BenchmarkResult {
  double grayscale_ms = 0.0;
  double blur_ms = 0.0;
  double edge_ms = 0.0;
  double total_ms = 0.0;
};

enum class EdgeMethod {
  Canny,
  Sobel
};

class ImageProcessor {
public:
  // Loads an image from disk (BGR). Throws std::runtime_error on failure.
  static cv::Mat loadImage(const std::string& path);

  // Saves an image to disk. Throws std::runtime_error on failure.
  static void saveImage(const std::string& path, const cv::Mat& image);

  // Runs: grayscale -> gaussian blur -> edge detection, with timings.
  // Returns edges image (single-channel) and fills benchmark timings.
  static cv::Mat processWithBenchmark(
      const cv::Mat& input_bgr,
      BenchmarkResult& out_bench,
      EdgeMethod method,
      int gaussian_kernel_size,
      double canny_low,
      double canny_high,
      int sobel_ksize);

private:
  static cv::Mat toGrayscale(const cv::Mat& input_bgr);
  static cv::Mat gaussianBlur(const cv::Mat& gray, int kernel_size);
  static cv::Mat cannyEdges(const cv::Mat& blurred, double low, double high);
  static cv::Mat sobelEdges(const cv::Mat& blurred, int ksize);

  static int makeOddAndMin3(int k);
};