cmake_minimum_required(VERSION 3.12)
project(Cheb2D)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
add_executable(main "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cxx")
if (MSVC)
target_sources(main PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
endif()
