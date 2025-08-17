project(PointMatch)

set(CONFIG_READER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "")

find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(configReader SHARED STATIC configReader.h configReader.cpp)
target_link_libraries (configReader ${PCL_LIBRARIES})
target_include_directories(configReader PUBLIC ${PCL_INCLUDE_DIRS})
