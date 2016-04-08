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

#include <boost/program_options.hpp>
#include <psl_base/exception.h>
#include <fstream>
#include <Eigen/Dense>
#include <psl_base/cameraMatrix.h>
#include <psl_stereo/cudaPlaneSweep.h>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include <iostream>

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
            ("dataFolder", boost::program_options::value<std::string>(&dataFolder)->default_value("DataPinholeCamera/niederdorf2"), "One of the data folders for pinhole planesweep provided with the plane sweep code.")
            ;

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).run(), vm);
    boost::program_options::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }

    std::string kMatrixFile = dataFolder + "/K.txt";

    // try to load k matrix
    std::ifstream kMatrixStr;
    kMatrixStr.open(kMatrixFile.c_str());

    if (!kMatrixStr.is_open())
    {
        PSL_THROW_EXCEPTION("Error opening K matrix file.")
    }

    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    kMatrixStr >> K(0,0);
    kMatrixStr >> K(0,1);
    kMatrixStr >> K(0,2);
    kMatrixStr >> K(1,1);
    kMatrixStr >> K(1,2);


    // Load the poses file
    // Poses are computed with Christopher Zach's publicly available V3D Software

    std::string v3dcameraFile = dataFolder + "/model-0-cams.txt";

    std::ifstream posesStr;
    posesStr.open(v3dcameraFile.c_str());

    if (!posesStr.is_open())
    {
        PSL_THROW_EXCEPTION("Could not load camera poses");
    }

    int numCameras;
    posesStr >> numCameras;

    std::map<int, PSL::CameraMatrix<double> > cameras;

    for (int c = 0; c < numCameras; c++)
    {
        int id;
        posesStr >> id;

        Eigen::Matrix<double, 3, 3> R;
        Eigen::Matrix<double, 3, 1> T;

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
                posesStr >> R(i,j);

            posesStr >> T(i);
        }

        cameras[id].setKRT(K, R, T);
    }

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

    // Each of the datasets contains 25 cameras taken in 5 rows
    // The reconstructions are not metric. In order to have an idea about the scale
    // everything is defined with respect to the average distance between the cameras.

    double avgDistance = 0;
    int numDistances = 0;

    for (unsigned int i = 0; i < imageFileNames.size()-1; i++)
    {
        if (cameras.count(i) == 1)
        {
            for (unsigned int j = i + 1; j < imageFileNames.size(); j++)
            {
                if (cameras.count(j) == 1)
                {
                    Eigen::Vector3d distance = cameras[i].getC() - cameras[j].getC();

                    avgDistance += distance.norm();
                    numDistances++;
                }
            }
        }
    }

    if (numDistances < 2)
    {
        PSL_THROW_EXCEPTION("Could not compute average distance, less than two cameras found");
    }

    avgDistance /= numDistances;

    float minZ = (float) (2.5f*avgDistance);
    float maxZ = (float) (100.0f*avgDistance);

    makeOutputFolder("pinholeTestResults");

    // First tests compute a depth map for the middle image of the first row
    {
        makeOutputFolder("pinholeTestResults/colorSAD");

        PSL::CudaPlaneSweep cPS;
        cPS.setScale(0.25); // Scale the images down to 0.25 times the original side length
        cPS.setZRange(minZ, maxZ);
        cPS.setMatchWindowSize(7,7);
        cPS.setNumPlanes(256);
        cPS.setOcclusionMode(PSL::PLANE_SWEEP_OCCLUSION_NONE);
        cPS.setPlaneGenerationMode(PSL::PLANE_SWEEP_PLANEMODE_UNIFORM_DISPARITY);
        cPS.setMatchingCosts(PSL::PLANE_SWEEP_SAD);
        cPS.setSubPixelInterpolationMode(PSL::PLANE_SWEEP_SUB_PIXEL_INTERP_INVERSE);
        cPS.enableOutputBestDepth();
        cPS.enableColorMatching();
        cPS.enableOutputBestCosts(false);
        cPS.enableOuputUniquenessRatio(false);
        cPS.enableOutputCostVolume(false);
        cPS.enableSubPixel();

        // now we upload the images
        int refId = -1;
        for (int i = 0; i < 5; i++)
        {
            // load the image from disk
            std::string imageFileName = dataFolder + "/" + imageFileNames[i];
            cv::Mat image = cv::imread(imageFileName);
            if (image.empty())
            {
                PSL_THROW_EXCEPTION("Failed to load image")
            }

            if (cameras.count(i) != 1)
            {
                PSL_THROW_EXCEPTION("Camera for image was not loaded, something is wrong with the dataset")
            }

            int id = cPS.addImage(image, cameras[i]);
            if (i == 2)
            {
                refId = id;
            }
        }

        {
            cPS.process(refId);
            PSL::DepthMap<float, double> dM;
            dM = cPS.getBestDepth();
            cv::Mat refImage = cPS.downloadImage(refId);

            makeOutputFolder("pinholeTestResults/colorSAD/NoOcclusionHandling/");
            cv::imwrite("pinholeTestResults/colorSAD/NoOcclusionHandling/refImg.png",refImage);
            dM.saveInvDepthAsColorImage("pinholeTestResults/colorSAD/NoOcclusionHandling/invDepthCol.png", minZ, maxZ);

            cv::imshow("Reference Image", refImage);
            dM.displayInvDepthColored(minZ, maxZ, 100);
        }

        {
            cPS.setOcclusionMode(PSL::PLANE_SWEEP_OCCLUSION_REF_SPLIT);
            cPS.process(refId);
            PSL::DepthMap<float, double> dM;
            dM = cPS.getBestDepth();
            cv::Mat refImage = cPS.downloadImage(refId);

            makeOutputFolder("pinholeTestResults/colorSAD/RefSplit/");
            cv::imwrite("pinholeTestResults/colorSAD/RefSplit/refImg.png",refImage);
            dM.saveInvDepthAsColorImage("pinholeTestResults/colorSAD/RefSplit/invDepthCol.png", minZ, maxZ);

            cv::imshow("Reference Image", refImage);
            dM.displayInvDepthColored(minZ, maxZ, 100);
        }


        // now we add the remaining images and use best K occlusion handling
        for (int i = 5; i < 25; i++)
        {
            // load the image from disk
            std::string imageFileName = dataFolder + "/" + imageFileNames[i];
            cv::Mat image = cv::imread(imageFileName);
            if (image.empty())
            {
                PSL_THROW_EXCEPTION("Failed to load image")
            }

            if (cameras.count(i) != 1)
            {
                PSL_THROW_EXCEPTION("Camera for image was not loaded, something is wrong with the dataset")
            }

            int id = cPS.addImage(image, cameras[i]);
            if (i == 12)
            {
                refId = id;
            }
        }

        {
            cPS.setOcclusionMode(PSL::PLANE_SWEEP_OCCLUSION_BEST_K);
            cPS.setOcclusionBestK(5);
            cPS.process(refId);
            PSL::DepthMap<float, double> dM;
            dM = cPS.getBestDepth();
            cv::Mat refImage = cPS.downloadImage(refId);

            makeOutputFolder("pinholeTestResults/colorSAD/BestK/");
            cv::imwrite("pinholeTestResults/colorSAD/BestK/refImg.png",refImage);
            dM.saveInvDepthAsColorImage("pinholeTestResults/colorSAD/BestK/invDepthCol.png", minZ, maxZ);

            cv::imshow("Reference Image", refImage);
            dM.displayInvDepthColored(minZ, maxZ, 100);

        }
    }

    // First tests compute a depth map for the middle image of the first row
    {
        makeOutputFolder("pinholeTestResults/grayscaleSAD");
        makeOutputFolder("pinholeTestResults/grayscaleZNCC");

        PSL::CudaPlaneSweep cPS;
        cPS.setScale(0.25); // Scale the images down to 0.25 times the original side length
        cPS.setZRange(minZ, maxZ);
        cPS.setMatchWindowSize(7,7);
        cPS.setNumPlanes(256);
        cPS.setOcclusionMode(PSL::PLANE_SWEEP_OCCLUSION_NONE);
        cPS.setPlaneGenerationMode(PSL::PLANE_SWEEP_PLANEMODE_UNIFORM_DISPARITY);
        cPS.setSubPixelInterpolationMode(PSL::PLANE_SWEEP_SUB_PIXEL_INTERP_INVERSE);
        cPS.enableOutputBestDepth();
        cPS.enableColorMatching(false);
        cPS.enableOutputBestCosts(false);
        cPS.enableOuputUniquenessRatio(false);
        cPS.enableOutputCostVolume(false);
        cPS.enableSubPixel();

        // now we upload the images
        int refId = -1;
        for (int i = 0; i < 5; i++)
        {
            // load the image from disk
            std::string imageFileName = dataFolder + "/" + imageFileNames[i];
            cv::Mat image = cv::imread(imageFileName);
            if (image.empty())
            {
                PSL_THROW_EXCEPTION("Failed to load image")
            }

            if (cameras.count(i) != 1)
            {
                PSL_THROW_EXCEPTION("Camera for image was not loaded, something is wrong with the dataset")
            }

            int id = cPS.addImage(image, cameras[i]);
            if (i == 2)
            {
                refId = id;
            }
        }

        {
            cPS.setMatchingCosts(PSL::PLANE_SWEEP_SAD);
            cPS.process(refId);
            PSL::DepthMap<float, double> dM;
            dM = cPS.getBestDepth();
            cv::Mat refImage = cPS.downloadImage(refId);

            makeOutputFolder("pinholeTestResults/grayscaleSAD/NoOcclusionHandling/");
            cv::imwrite("pinholeTestResults/grayscaleSAD/NoOcclusionHandling/refImg.png",refImage);
            dM.saveInvDepthAsColorImage("pinholeTestResults/grayscaleSAD/NoOcclusionHandling/invDepthCol.png", minZ, maxZ);

            cv::imshow("Reference Image", refImage);
            dM.displayInvDepthColored(minZ, maxZ, 100);
        }

        {
            cPS.setMatchingCosts(PSL::PLANE_SWEEP_ZNCC);
            cPS.process(refId);
            PSL::DepthMap<float, double> dM;
            dM = cPS.getBestDepth();
            cv::Mat refImage = cPS.downloadImage(refId);

            makeOutputFolder("pinholeTestResults/grayscaleZNCC/NoOcclusionHandling/");
            cv::imwrite("pinholeTestResults/grayscaleZNCC/NoOcclusionHandling/refImg.png",refImage);
            dM.saveInvDepthAsColorImage("pinholeTestResults/grayscaleZNCC/NoOcclusionHandling/invDepthCol.png", minZ, maxZ);

            cv::imshow("Reference Image", refImage);
            dM.displayInvDepthColored(minZ, maxZ, 100);
        }

        cPS.setOcclusionMode(PSL::PLANE_SWEEP_OCCLUSION_REF_SPLIT);

        {
            cPS.setMatchingCosts(PSL::PLANE_SWEEP_SAD);
            cPS.process(refId);
            PSL::DepthMap<float, double> dM;
            dM = cPS.getBestDepth();
            cv::Mat refImage = cPS.downloadImage(refId);

            makeOutputFolder("pinholeTestResults/grayscaleSAD/RefSplit/");
            cv::imwrite("pinholeTestResults/grayscaleSAD/RefSplit/refImg.png",refImage);
            dM.saveInvDepthAsColorImage("pinholeTestResults/grayscaleSAD/RefSplit/invDepthCol.png", minZ, maxZ);

            cv::imshow("Reference Image", refImage);
            dM.displayInvDepthColored(minZ, maxZ, 100);
        }

        {
            cPS.setMatchingCosts(PSL::PLANE_SWEEP_ZNCC);
            cPS.process(refId);
            PSL::DepthMap<float, double> dM;
            dM = cPS.getBestDepth();
            cv::Mat refImage = cPS.downloadImage(refId);

            makeOutputFolder("pinholeTestResults/grayscaleZNCC/RefSplit/");
            cv::imwrite("pinholeTestResults/grayscaleZNCC/RefSplit/refImg.png",refImage);
            dM.saveInvDepthAsColorImage("pinholeTestResults/grayscaleZNCC/RefSplit/invDepthCol.png", minZ, maxZ);

            cv::imshow("Reference Image", refImage);
            dM.displayInvDepthColored(minZ, maxZ, 100);
        }


        // now we add the remaining images and use best K occlusion handling
        for (int i = 5; i < 25; i++)
        {
            // load the image from disk
            std::string imageFileName = dataFolder + "/" + imageFileNames[i];
            cv::Mat image = cv::imread(imageFileName);
            if (image.empty())
            {
                PSL_THROW_EXCEPTION("Failed to load image")
            }

            if (cameras.count(i) != 1)
            {
                PSL_THROW_EXCEPTION("Camera for image was not loaded, something is wrong with the dataset")
            }

            int id = cPS.addImage(image, cameras[i]);
            if (i == 12)
            {
                refId = id;
            }
        }

        cPS.setOcclusionMode(PSL::PLANE_SWEEP_OCCLUSION_BEST_K);

        {
            cPS.setMatchingCosts(PSL::PLANE_SWEEP_SAD);
            cPS.setOcclusionBestK(5);
            cPS.process(refId);
            PSL::DepthMap<float, double> dM;
            dM = cPS.getBestDepth();
            cv::Mat refImage = cPS.downloadImage(refId);

            makeOutputFolder("pinholeTestResults/grayscaleSAD/BestK/");
            cv::imwrite("pinholeTestResults/grayscaleSAD/BestK/refImg.png",refImage);
            dM.saveInvDepthAsColorImage("pinholeTestResults/grayscaleSAD/BestK/invDepthCol.png", minZ, maxZ);

            cv::imshow("Reference Image", refImage);
            dM.displayInvDepthColored(minZ, maxZ, 100);
        }

        {
            cPS.setMatchingCosts(PSL::PLANE_SWEEP_ZNCC);
            cPS.setOcclusionBestK(5);
            cPS.process(refId);
            PSL::DepthMap<float, double> dM;
            dM = cPS.getBestDepth();
            cv::Mat refImage = cPS.downloadImage(refId);

            makeOutputFolder("pinholeTestResults/grayscaleZNCC/BestK/");
            cv::imwrite("pinholeTestResults/grayscaleZNCC/BestK/refImg.png",refImage);
            dM.saveInvDepthAsColorImage("pinholeTestResults/grayscaleZNCC/BestK/invDepthCol.png", minZ, maxZ);

            cv::imshow("Reference Image", refImage);
            dM.displayInvDepthColored(minZ, maxZ, 100);
        }
    }




}
