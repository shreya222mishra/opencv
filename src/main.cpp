#include "image_processor.h"

#include <iostream>
#include <string>
#include <vector>

static void printUsage(const char* argv0) {
  std::cout
    << "Usage:\n"
    << "  " << argv0 << " --input <path> --output <path> [options]\n\n"
    << "Options:\n"
    << "  --method canny|sobel        Edge method (default: canny)\n"
    << "  --blur <odd_int>=5          Gaussian kernel size (default: 5)\n"
    << "  --canny-low <double>=50     Canny low threshold (default: 50)\n"
    << "  --canny-high <double>=150   Canny high threshold (default: 150)\n"
    << "  --sobel-ksize <odd_int>=3   Sobel kernel size (default: 3)\n"
    << "  --repeat <int>=1            Repeat processing N times for benchmarking\n";
}

static bool getArgValue(const std::vector<std::string>& args, const std::string& key, std::string& out) {
  for (size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == key) {
      out = args[i + 1];
      return true;
    }
  }
  return false;
}

int main(int argc, char** argv) {
  try {
    std::vector<std::string> args(argv + 1, argv + argc);

    std::string input_path, output_path;
    if (!getArgValue(args, "--input", input_path) || !getArgValue(args, "--output", output_path)) {
      printUsage(argv[0]);
      return 1;
    }

    std::string method_str = "canny";
    (void)getArgValue(args, "--method", method_str);

    int blur_k = 5;
    std::string blur_str;
    if (getArgValue(args, "--blur", blur_str)) blur_k = std::stoi(blur_str);

    double canny_low = 50.0, canny_high = 150.0;
    std::string low_str, high_str;
    if (getArgValue(args, "--canny-low", low_str)) canny_low = std::stod(low_str);
    if (getArgValue(args, "--canny-high", high_str)) canny_high = std::stod(high_str);

    int sobel_ksize = 3;
    std::string sobel_str;
    if (getArgValue(args, "--sobel-ksize", sobel_str)) sobel_ksize = std::stoi(sobel_str);

    int repeat = 1;
    std::string rep_str;
    if (getArgValue(args, "--repeat", rep_str)) repeat = std::max(1, std::stoi(rep_str));

    EdgeMethod method = (method_str == "sobel") ? EdgeMethod::Sobel : EdgeMethod::Canny;

    cv::Mat input = ImageProcessor::loadImage(input_path);

    // Run N times and average timings (helps reduce noise)
    BenchmarkResult sum{};
    cv::Mat edges;

    for (int i = 0; i < repeat; ++i) {
      BenchmarkResult bench{};
      edges = ImageProcessor::processWithBenchmark(
          input, bench, method, blur_k, canny_low, canny_high, sobel_ksize);

      sum.grayscale_ms += bench.grayscale_ms;
      sum.blur_ms += bench.blur_ms;
      sum.edge_ms += bench.edge_ms;
      sum.total_ms += bench.total_ms;
    }

    BenchmarkResult avg{};
    avg.grayscale_ms = sum.grayscale_ms / repeat;
    avg.blur_ms = sum.blur_ms / repeat;
    avg.edge_ms = sum.edge_ms / repeat;
    avg.total_ms = sum.total_ms / repeat;

    ImageProcessor::saveImage(output_path, edges);

    std::cout << "Saved edges to: " << output_path << "\n";
    std::cout << "Benchmark (avg over " << repeat << " run(s)):\n";
    std::cout << "  grayscale: " << avg.grayscale_ms << " ms\n";
    std::cout << "  blur:      " << avg.blur_ms << " ms\n";
    std::cout << "  edge:      " << avg.edge_ms << " ms\n";
    std::cout << "  total:     " << avg.total_ms << " ms\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }
}