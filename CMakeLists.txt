cmake_minimum_required(VERSION 3.16)
project(edge_benchmark_opencv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(edge_benchmark
  src/main.cpp
  src/image_processor.cpp
)

target_include_directories(edge_benchmark PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(edge_benchmark PRIVATE ${OpenCV_LIBS})
