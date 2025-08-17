project(PointMatch)

add_executable(PointMatch main.cpp)

target_link_libraries(PointMatch PPF configReader)
target_include_directories(PointMatch PUBLIC ${PCL_INCLUDE_DIRS} ${PCLFUNCTION_INCLUDE_DIR} ${PPF_INCLUDE_DIR} ${CONFIG_READER_INCLUDE_DIR})

