Content and Copyright
--------------------------------

This software package PlaneSweepLib (PSL) contains a C++/CUDA implementation of plane sweeping stereo matching for pinhole and fisheye images.
The package comes with test data and scripts to run the software on the test data.
It was tested on Linux using the GCC Toolchain and on Windows using Visual Studio.

Copyright 2016 Christian Haene (ETH Zuerich)

PSL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PSL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PSL.  If not, see <http://www.gnu.org/licenses/>.


Version
--------------------------------

1.0 - February 24 2016


Citation
--------------------------------

If you use this software package please cite the relevant publications.

For using the plane sweep code please cite:

Christian Häne, Lionel Heng, Gim Hee Lee, Alexey Sizov, Marc Pollefeys,
Real-Time Direct Dense Matching on Fisheye Images Using Plane-Sweeping Stereo,
Proc Int. Conf. on 3D Vison (3DV) 2014

The datasets provided with the pinhole version of the code were initially published in:

Christian Häne, Christopher Zach, Bernhard Zeisl, Marc Pollefeys,
A Patch Prior for Dense 3D Reconstruction in Man-Made Environments,
Proc. Int. Conf. on 3D Data, Imaging, Modeling, Processing, Visualization and Transmission (3DIMPVT) 2012


Furthermore this code served as a basis for the papers:

Christian Häne, Torsten Sattler, Marc Pollefeys,
Obstacle Detection for Self-Driving Cars Using Only Monocular Cameras and Wheel Odometry,
Proc. IEEE/RSJ Int. Conf. on Intelligent Robots and Systems (IROS) 2015

Thomas Schöps, Torsten Sattler, Christin Häne, Marc Pollefeys,
3D Modeling on the Go: Interactive 3D Reconstruction of Large-Scale Scenes on Mobile Devices,
Proc Int. Conf. on 3D Vison (3DV) 2015


Prerequisites
--------------------------------

The code is written in C++ and CUDA. To use it you need a CUDA compatible Nvidia GPU.
The following libraries are required:

GCC Toolchain on Linux
Visual Studio C++ on Windows
CMake
Nvidia CUDA
Boost (system filesystem program_options)
OpenCV
Eigen3


Instructions Linux
-----------------------------

To compile the code open a terminal and cd to the root directory PlaneSweepLib of the package and run:

mkdir build
cd build
cmake ..
make

The package comes with test data and applications that show how to use the plane sweep code.
To run the tests on the provided data cd to the root directory PlaneSweepLib of the package and run:

sh runPinholePlanesweepTestsLinux.sh
sh runFisheyePlanesweepTestsLinux.sh

The results are written to the folder PlaneSweepLib/testResults.
To check if the code runs correctly reference results are provided in the folder PlaneSweepLib/referenceTestResults


Instructions Windows
---------------------------------

Open a command prompt and cd to the root directory PlaneSweepLib of the package and run:

mkdir build
cd build
cmake -G "Visual Studio 10 Win64" ..    (Replace "Visual Studio 10 Win64" with your version, cmake without arguments provides a list.
                                        Alternatively use cmake-gui to generate the project files.)
                                             
Open the project file PlaneSweepLib/build/PSL.sln with Visual Studio and compile the whole solution in Release mode.


The package comes with test data and applications that show how to use the plane sweep code.
To run the tests on the provided data go to the command prompt and cd to the root directory PlaneSweepLib of the package and run:

runPinholePlanesweepTestsWindows.bat
runFisheyePlanesweepTestsWindows.bat

The results are written to the folder PlaneSweepLib/testResults.
To check if the code runs correctly reference results are provided in the folder PlaneSweepLib/referenceTestResults


Acknowledgements
----------------------------------

This code was written in the Computer Vision and Geometry Group (CVG) led by Prof. Marc Pollefeys.
Helpful discussions and suggestions from Thomas Schöps helped to improve the quality and performance of the code.
The code was written as part of the V-Charge project, grant #269916 under the European Community’s Seventh Framework Programme (FP7/2007-2013)



