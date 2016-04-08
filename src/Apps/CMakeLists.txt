find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

add_executable(cudaPlanesweepMAPTk cudaPlanesweepMAPTk.cpp)
target_link_libraries(cudaPlanesweepMAPTk PSL ${VTK_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})
