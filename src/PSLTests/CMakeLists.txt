link_libraries(PSL ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(pinholePlanesweepTest pinholePlanesweepTest.cpp)
add_executable(fisheyePlanesweepTest fisheyePlanesweepTest.cpp)

