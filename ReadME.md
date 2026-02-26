# OpenCV Edge Detection + Performance Benchmark (C++)

A small, modular C++ project using OpenCV that runs a simple image-processing pipeline and benchmarks runtime using `std::chrono`.

## Features
- Load image from disk
- Convert to grayscale
- Apply Gaussian blur
- Run edge detection (**Canny** or **Sobel**)
- Benchmark each stage + total runtime (averaged with `--repeat`)
- Save the output edge map

## Project Structure


├── CMakeLists.txt
├── ReadMe.md
├── include
│ └── image_processor.h
└── src
├── image_processor.cpp
└── main.cpp


## Requirements
- C++17 compiler (Apple Clang / g++ / clang++)
- CMake (3.16+ recommended)
- OpenCV 4.x

### Install (macOS - Homebrew)
brew update
brew install cmake opencv

Install (Ubuntu/Debian)
sudo apt update
sudo apt install -y cmake g++ libopencv-dev
Build

From the project root:

mkdir -p build
cd build
cmake ..
cmake --build . -j
If CMake cannot find OpenCV (common on macOS/Homebrew)
cmake .. -DOpenCV_DIR=$(brew --prefix opencv)/lib/cmake/opencv4
cmake --build . -j
Run

Place an image in the project root (example: input.jpg) and run:

Canny (default)
./edge_benchmark --input ../input.jpg --output ../edges.png
