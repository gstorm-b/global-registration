cmake_minimum_required(VERSION 3.0)

project(PointMatch
  VERSION 0.1.0
  DESCRIPTION "PointMatch: match point cloud"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# add_subdirectory(third_party)
add_subdirectory(include)
add_subdirectory(apps)
