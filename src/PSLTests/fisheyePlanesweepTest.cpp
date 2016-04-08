// This file is part of PlaneSweepLib (PSL)

// Copyright 2016 Christian Haene (ETH Zuerich)

// PSL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// PSL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with PSL.  If not, see <http://www.gnu.org/licenses/>.
#include <string>
#include <boost/program_options.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <psl_base/exception.h>
#include <opencv2/highgui/highgui.hpp>
#include <psl_cudaBase/cudaFishEyeImageProcessor.h>
#include <psl_stereo/cudaFishEyePlaneSweep.h>
#include <boost/filesystem.hpp>
#include <cmath>
#include <cstdlib>
#include <psl_base/common.h>

void makeOutputFolder(std::string folderName)
{
    if (!boost::filesystem::exists(folderName))
    {
        if (!boost::filesystem::create_directory(folderName))
        {
            std::stringstream errorMsg;
            errorMsg << "Could not create output directory: " << folderName;
            PSL_THROW_EXCEPTION(errorMsg.str().c_str());
        }
    }
}

int main(int argc, char* argv[])
{
    std::string dataFolder;

    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
            ("help", "Produce help message")
            ("dataFolder", boost::program_options::value<std::string>(&dataFolder)->default_value("DataFisheyeCamera/right"), "One of the data folders for pinhole planesweep provided with the plane sweep code.")
            ;

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).run(), vm);
    boost::program_options::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }

    // read in the calibration
    std::string calibFileName = dataFolder + "/calib.txt";
    double xi;
    double k1, k2, p1, p2;
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    Eigen::Vector3d C;
    Eigen::Matrix3d R;

    std::ifstream calibrationStr;
    calibrationStr.open(calibFileName.c_str());

    if (!calibrationStr.is_open())
    {
        PSL_THROW_EXCEPTION("Error opening calibration file calib.txt.")
    }

    // intrinsic calibration and distortion parameters
    calibrationStr >> K(0,0); calibrationStr >> K(0,1); calibrationStr >> K(0,2);
    calibrationStr >> K(1,1); calibrationStr >> K(1,2);
    calibrationStr >> xi;
    calibrationStr >> k1; calibrationStr >> k2; calibrationStr >> p1; calibrationStr >> p2;

    // extrinsic calibration
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            calibrationStr >> R(i,j);

        }
        calibrationStr >> C(i);
    }

   // std::cout << R << std::endl;
   // std::cout << C << std::endl;

    // now load the image filenames
    std::string imageListFile = dataFolder + "/images.txt";

    std::ifstream imagesStream;
    imagesStream.open(imageListFile.c_str());

    if (!imagesStream.is_open())
    {
        PSL_THROW_EXCEPTION("Could not load images list file")
    }

    std::vector<std::string> imageFileNames;
    {
        std::string imageFileName;
        while (imagesStream >> imageFileName)
        {
            imageFileNames.push_back(imageFileName);
        }
    }

    if (imageFileNames.size() != 25)
    {
        PSL_THROW_EXCEPTION("The dataset does not contain 25 images")
    }

    // display the images for a test
//    for (int i = 0; i < imageFileNames.size(); i++)
//    {
//        std::string imageFileName = dataFolder + "/" + imageFileNames[i];
//        cv::Mat image = cv::imread(imageFileName);
//        cv::imshow("image", image);
//        cv::waitKey(50);
//    }


    // reading poses from file
    // read in the poses
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> > systemR;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > systemT;
    std::vector<uint64_t> timestamps;

    std::ifstream systemPosesFile;
    systemPosesFile.open((dataFolder + "/system_poses.txt").c_str());
    if (!systemPosesFile.is_open())
    {
        PSL_THROW_EXCEPTION("Error opening system_poses.txt");
    }


    std::string line;
    while (std::getline(systemPosesFile, line))
    {
        if (line[0] == '#')
            continue;

        std::stringstream lineStr(line);

        uint64_t timestamp;

        double yaw, pitch, roll, t_x, t_y, t_z;
        lineStr >> timestamp >> yaw >> pitch >> roll >> t_x >> t_y >> t_z;

        Eigen::Matrix3d rotation;
        rotation = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
                * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
                * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

        systemR.push_back(rotation);

        Eigen::Vector3d T(t_x, t_y, t_z);
        systemT.push_back(T);

        timestamps.push_back(timestamp);
    }

    PSL_CUDA::DeviceImage devImg;
    PSL_CUDA::CudaFishEyeImageProcessor cFEIP;

    double minZ = 0.4;
    double maxZ = 5000;

    makeOutputFolder("fisheyeTestResults");

    // First tests use the first 5 images
    {

        PSL::CudaFishEyePlaneSweep cFEPS;
        cFEPS.setScale(1.0);
        cFEPS.setZRange(minZ, maxZ);
        cFEPS.setMatchWindowSize(7,7);
        cFEPS.setNumPlanes(256);
        cFEPS.setOcclusionMode(PSL::FISH_EYE_PLANE_SWEEP_OCCLUSION_NONE);
        cFEPS.setPlaneGenerationMode(PSL::FISH_EYE_PLANE_SWEEP_PLANEMODE_UNIFORM_DISPARITY);
        cFEPS.setMatchingCosts(PSL::FISH_EYE_PLANE_SWEEP_SAD);
        cFEPS.setSubPixelInterpolationMode(PSL::FISH_EYE_PLANE_SWEEP_SUB_PIXEL_INTERP_INVERSE);
        cFEPS.enableOutputBestDepth();
        cFEPS.enableOutputBestCosts(false);
        cFEPS.enableOuputUniquenessRatio(false);
        cFEPS.enableOutputCostVolume(false);
        cFEPS.enableSubPixel();

        // undistort and add the images
        int refId = -1;
        for (unsigned int i = 0; i < 5; i++)
        {

            std::string timestampStr = imageFileNames[i].substr(imageFileNames[i].length() - 20, 16);
            uint64_t timestamp = std::atoll(timestampStr.c_str());

            if (timestamp != timestamps[i])
            {
                PSL_THROW_EXCEPTION("Sequence broken, timestamps of poses and images do not match.")
            }

            std::string imageFileName = dataFolder + "/" + imageFileNames[i];
            cv::Mat imageOrig = cv::imread(imageFileName, 0);


            if (imageOrig.empty())
            {
                PSL_THROW_EXCEPTION("Error loading image.")
            }

            devImg.allocatePitchedAndUpload(imageOrig);

            // Assemble camera matrix
            Eigen::Matrix4d cameraToSystem = Eigen::Matrix4d::Identity();
            cameraToSystem.topLeftCorner(3,3) = R;
            cameraToSystem.topRightCorner(3,1) = C;

            Eigen::Matrix4d systemToWorld = Eigen::Matrix4d::Identity();
            systemToWorld.topLeftCorner(3,3) = systemR[i];
            systemToWorld.topRightCorner(3,1) = systemT[i];

            Eigen::Matrix4d worldToCamera = cameraToSystem.inverse()*systemToWorld.inverse();

            PSL::FishEyeCameraMatrix<double> cam(K, worldToCamera.topLeftCorner(3,3), worldToCamera.topRightCorner(3,1), xi);

            cFEIP.setInputImg(devImg, cam);

            std::pair<PSL_CUDA::DeviceImage, PSL::FishEyeCameraMatrix<double > > undistRes = cFEIP.undistort(0.5, 1.0, k1, k2, p1, p2);

            int id = cFEPS.addDeviceImage(undistRes.first, undistRes.second);

            if (i == 2)
            {
                refId = id;
            }
        }

        makeOutputFolder("fisheyeTestResults/grayscaleSAD");

        {
            cFEPS.process(refId);
            PSL::FishEyeDepthMap<float, double> fEDM;
            fEDM = cFEPS.getBestDepth();
            cv::Mat refImage = cFEPS.downloadImage(refId);

            makeOutputFolder("fisheyeTestResults/grayscaleSAD/NoOcclusionHandling/");
            cv::imwrite("fisheyeTestResults/grayscaleSAD/NoOcclusionHandling/refImg.png",refImage);
            fEDM.saveInvDepthAsColorImage("fisheyeTestResults/grayscaleSAD/NoOcclusionHandling/invDepthCol.png", (float) minZ, (float) maxZ);

            cv::imshow("Reference Image", refImage);
            fEDM.displayInvDepthColored((float) minZ, (float) maxZ, 100);
        }

        {
            cFEPS.setOcclusionMode(PSL::FISH_EYE_PLANE_SWEEP_OCCLUSION_REF_SPLIT);
            cFEPS.process(refId);
            PSL::FishEyeDepthMap<float, double> fEDM;
            fEDM = cFEPS.getBestDepth();
            cv::Mat refImage = cFEPS.downloadImage(refId);

            makeOutputFolder("fisheyeTestResults/grayscaleSAD/RefSplit/");
            cv::imwrite("fisheyeTestResults/grayscaleSAD/RefSplit/refImg.png",refImage);
            fEDM.saveInvDepthAsColorImage("fisheyeTestResults/grayscaleSAD/RefSplit/invDepthCol.png", (float) minZ, (float) maxZ);

            cv::imshow("Reference Image", refImage);
            fEDM.displayInvDepthColored((float) minZ, (float) maxZ, 100);
        }

        makeOutputFolder("fisheyeTestResults/grayscaleZNCC");

        {
            cFEPS.setMatchingCosts(PSL::FISH_EYE_PLANE_SWEEP_ZNCC);
            cFEPS.setOcclusionMode(PSL::FISH_EYE_PLANE_SWEEP_OCCLUSION_NONE);
            cFEPS.process(refId);
            PSL::FishEyeDepthMap<float, double> fEDM;
            fEDM = cFEPS.getBestDepth();
            cv::Mat refImage = cFEPS.downloadImage(refId);

            makeOutputFolder("fisheyeTestResults/grayscaleZNCC/NoOcclusionHandling/");
            cv::imwrite("fisheyeTestResults/grayscaleZNCC/NoOcclusionHandling/refImg.png",refImage);
            fEDM.saveInvDepthAsColorImage("fisheyeTestResults/grayscaleZNCC/NoOcclusionHandling/invDepthCol.png", (float) minZ, (float) maxZ);

            cv::imshow("Reference Image", refImage);
            fEDM.displayInvDepthColored((float) minZ, (float) maxZ, 100);
        }

        {
            cFEPS.setOcclusionMode(PSL::FISH_EYE_PLANE_SWEEP_OCCLUSION_REF_SPLIT);
            cFEPS.process(refId);
            PSL::FishEyeDepthMap<float, double> fEDM;
            fEDM = cFEPS.getBestDepth();
            cv::Mat refImage = cFEPS.downloadImage(refId);

            makeOutputFolder("fisheyeTestResults/grayscaleZNCC/RefSplit/");
            cv::imwrite("fisheyeTestResults/grayscaleZNCC/RefSplit/refImg.png",refImage);
            fEDM.saveInvDepthAsColorImage("fisheyeTestResults/grayscaleZNCC/RefSplit/invDepthCol.png", (float) minZ, (float) maxZ);

            cv::imshow("Reference Image", refImage);
            fEDM.displayInvDepthColored((float) minZ, (float) maxZ, 100);
        }

        // now the remaining images are added and best K occlusion handling is performed
        for (unsigned int i = 5; i < 25; i++)
        {

            std::string timestampStr = imageFileNames[i].substr(imageFileNames[i].length() - 20, 16);
            uint64_t timestamp = std::atoll(timestampStr.c_str());

            if (timestamp != timestamps[i])
            {
                PSL_THROW_EXCEPTION("Sequence broken, timestamps of poses and images do not match.")
            }

            std::string imageFileName = dataFolder + "/" + imageFileNames[i];
            cv::Mat imageOrig = cv::imread(imageFileName, 0);


            if (imageOrig.empty())
            {
                PSL_THROW_EXCEPTION("Error loading image.")
            }

            devImg.allocatePitchedAndUpload(imageOrig);

            // Assemble camera matrix
            Eigen::Matrix4d cameraToSystem = Eigen::Matrix4d::Identity();
            cameraToSystem.topLeftCorner(3,3) = R;
            cameraToSystem.topRightCorner(3,1) = C;

            Eigen::Matrix4d systemToWorld = Eigen::Matrix4d::Identity();
            systemToWorld.topLeftCorner(3,3) = systemR[i];
            systemToWorld.topRightCorner(3,1) = systemT[i];

            Eigen::Matrix4d worldToCamera = cameraToSystem.inverse()*systemToWorld.inverse();

            PSL::FishEyeCameraMatrix<double> cam(K, worldToCamera.topLeftCorner(3,3), worldToCamera.topRightCorner(3,1), xi);

            cFEIP.setInputImg(devImg, cam);

            std::pair<PSL_CUDA::DeviceImage, PSL::FishEyeCameraMatrix<double > > undistRes = cFEIP.undistort(0.5, 1.0, k1, k2, p1, p2);

            int id = cFEPS.addDeviceImage(undistRes.first, undistRes.second);

            if (i == 12)
            {
                refId = id;
            }
        }

        {
            cFEPS.setMatchingCosts(PSL::FISH_EYE_PLANE_SWEEP_SAD);
            cFEPS.setOcclusionMode(PSL::FISH_EYE_PLANE_SWEEP_OCCLUSION_BEST_K);
            cFEPS.setOcclusionBestK(5);

            cFEPS.process(refId);
            PSL::FishEyeDepthMap<float, double> fEDM;
            fEDM = cFEPS.getBestDepth();
            cv::Mat refImage = cFEPS.downloadImage(refId);

            makeOutputFolder("fisheyeTestResults/grayscaleSAD/BestK/");
            cv::imwrite("fisheyeTestResults/grayscaleSAD/BestK/refImg.png",refImage);
            fEDM.saveInvDepthAsColorImage("fisheyeTestResults/grayscaleSAD/BestK/invDepthCol.png", (float) minZ, (float) maxZ);

            cv::imshow("Reference Image", refImage);
            fEDM.displayInvDepthColored((float) minZ, (float) maxZ, 100);
        }

        {
            cFEPS.setMatchingCosts(PSL::FISH_EYE_PLANE_SWEEP_ZNCC);

            cFEPS.process(refId);
            PSL::FishEyeDepthMap<float, double> fEDM;
            fEDM = cFEPS.getBestDepth();
            cv::Mat refImage = cFEPS.downloadImage(refId);

            makeOutputFolder("fisheyeTestResults/grayscaleZNCC/BestK/");
            cv::imwrite("fisheyeTestResults/grayscaleZNCC/BestK/refImg.png",refImage);
            fEDM.saveInvDepthAsColorImage("fisheyeTestResults/grayscaleZNCC/BestK/invDepthCol.png", (float) minZ, (float) maxZ);

            cv::imshow("Reference Image", refImage);
            fEDM.displayInvDepthColored((float) minZ, (float) maxZ, 100);
        }


    }


}
