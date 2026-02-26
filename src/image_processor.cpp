#include "image_processor.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <stdexcept>

namespace {
using Clock = std::chrono::high_resolution_clock;

double msSince(const Clock::time_point& start, const Clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}
} // namespace

cv::Mat ImageProcessor::loadImage(const std::string& path) {
  cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
  if (img.empty()) {
    throw std::runtime_error("Failed to load image: " + path);
  }
  return img;
}

void ImageProcessor::saveImage(const std::string& path, const cv::Mat& image) {
  if (image.empty()) {
    throw std::runtime_error("Cannot save an empty image: " + path);
  }
  if (!cv::imwrite(path, image)) {
    throw std::runtime_error("Failed to save image: " + path);
  }
}

cv::Mat ImageProcessor::processWithBenchmark(
    const cv::Mat& input_bgr,
    BenchmarkResult& out_bench,
    EdgeMethod method,
    int gaussian_kernel_size,
    double canny_low,
    double canny_high,
    int sobel_ksize) {

  if (input_bgr.empty()) {
    throw std::runtime_error("Input image is empty.");
  }

  const auto t0 = Clock::now();

  const auto g0 = Clock::now();
  cv::Mat gray = toGrayscale(input_bgr);
  const auto g1 = Clock::now();
  out_bench.grayscale_ms = msSince(g0, g1);

  const auto b0 = Clock::now();
  cv::Mat blurred = gaussianBlur(gray, gaussian_kernel_size);
  const auto b1 = Clock::now();
  out_bench.blur_ms = msSince(b0, b1);

  const auto e0 = Clock::now();
  cv::Mat edges;
  if (method == EdgeMethod::Canny) {
    edges = cannyEdges(blurred, canny_low, canny_high);
  } else {
    edges = sobelEdges(blurred, sobel_ksize);
  }
  const auto e1 = Clock::now();
  out_bench.edge_ms = msSince(e0, e1);

  const auto t1 = Clock::now();
  out_bench.total_ms = msSince(t0, t1);

  return edges;
}

cv::Mat ImageProcessor::toGrayscale(const cv::Mat& input_bgr) {
  cv::Mat gray;
  cv::cvtColor(input_bgr, gray, cv::COLOR_BGR2GRAY);
  return gray;
}

int ImageProcessor::makeOddAndMin3(int k) {
  if (k < 3) k = 3;
  if (k % 2 == 0) ++k;
  return k;
}

cv::Mat ImageProcessor::gaussianBlur(const cv::Mat& gray, int kernel_size) {
  if (gray.empty()) {
    throw std::runtime_error("Grayscale image is empty.");
  }
  kernel_size = makeOddAndMin3(kernel_size);

  cv::Mat out;
  cv::GaussianBlur(gray, out, cv::Size(kernel_size, kernel_size), 0.0);
  return out;
}

cv::Mat ImageProcessor::cannyEdges(const cv::Mat& blurred, double low, double high) {
  if (blurred.empty()) {
    throw std::runtime_error("Blurred image is empty.");
  }
  if (low < 0 || high < 0 || high < low) {
    throw std::runtime_error("Invalid Canny thresholds.");
  }

  cv::Mat edges;
  cv::Canny(blurred, edges, low, high);
  return edges;
}

cv::Mat ImageProcessor::sobelEdges(const cv::Mat& blurred, int ksize) {
  if (blurred.empty()) {
    throw std::runtime_error("Blurred image is empty.");
  }
  ksize = makeOddAndMin3(ksize);

  cv::Mat grad_x, grad_y;
  cv::Sobel(blurred, grad_x, CV_16S, 1, 0, ksize);
  cv::Sobel(blurred, grad_y, CV_16S, 0, 1, ksize);

  cv::Mat abs_x, abs_y;
  cv::convertScaleAbs(grad_x, abs_x);
  cv::convertScaleAbs(grad_y, abs_y);

  cv::Mat edges;
  cv::addWeighted(abs_x, 0.5, abs_y, 0.5, 0.0, edges);
  return edges;
}