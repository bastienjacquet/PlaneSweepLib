#include <boost/filesystem.hpp>
#include <boost/timer.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <psl_base/cameraMatrix.h>
#include <psl_stereo/cudaPlaneSweep.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPolyData.h>
#include <vtkNew.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkStructuredGrid.h>
#include <vtkXMLStructuredGridWriter.h>
#include <vtkImageData.h>
#include <vtkXMLImageDataWriter.h>

bool overlapCompare(std::pair<int, float> p1, std::pair<int, float> p2)
{
    return p1.second > p2.second;
}
double cameraDirAngleCosine(PSL::CameraMatrix<double>  const & cam1,PSL::CameraMatrix<double> const & cam2){
    Eigen::Vector4d dirHomo = cam1.unprojectPoint(cam1.getK()(0,2),cam1.getK()(1,2),1);
    dirHomo/=dirHomo(3);
    Eigen::Vector3d dir1(dirHomo(0),dirHomo(1),dirHomo(2));
    dir1 -=  cam1.getC();
    dir1.normalize();

    Eigen::Vector4d dirHomo2 = cam2.unprojectPoint(cam2.getK()(0,2),cam2.getK()(1,2),1);
    dirHomo2/=dirHomo2(3);
    Eigen::Vector3d dir2(dirHomo2(0),dirHomo2(1),dirHomo2(2));
    dir2 -= cam2.getC();
    dir2.normalize();

    return dir2.dot(dir1);
}

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
    bool display = false;
    std::string frameFolder;
    std::string krtdFolder;
    std::string landmarksPLY;
    std::string outDir;
    std::string configFile;
    std::string imageListFile;
    int view;
    bool debug = true;
    bool filter = false;
    bool storeBestCostsAndUniquenessRatios = true;
    bool krtdOnly = false;
//    bool writePointClouds = false;
    std::string writePointClouds;

    // pose grouping parameters
    int pGnumPlanes;
    float pGScale;
    float pGMinDepth;
    float pGMaxDepth;

    // plane sweep parameters
    bool pSAutoRange;
    float pSMinDepth=0,pSMaxDepth=0;
//    if (!pSAutoRange){
//        pSMinDepth = conf.getAsFloat("PS_MIN_DEPTH",true);
//        pSMaxDepth = conf.getAsFloat("PS_MAX_DEPTH",true);
//    }
    int pSNumPlanes;
    float pSRescaleFactor;

    std::string pSMatchingCostsStr;

    int pSMatchingWindowSize;

    bool pSColorMatching;

    std::string pSOcclusionModeStr;
    int pSOcclusionBestKK;

    bool psEnableSubPixel;

    int pSMaxNumImages;

    float costThreshold;
    float thresholdUniq;

    float minAngleDegree;

    int refViewStep;

    std::string fileType;

    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
            ("help", "Produce help message")
            ("frameFolder", boost::program_options::value<std::string>(&frameFolder)->default_value("capture_1/"), "Folders containing the frames.")
            ("krtdFolder", boost::program_options::value<std::string>(&krtdFolder)->default_value("ba_output/krtd/"), "Folders containing the .krtd files.")
            ("landmarksPLY", boost::program_options::value<std::string>(&landmarksPLY)->default_value("ba_output/landmarks.ply"), "File containing the landmark points (to estimate where dense estimation should take place).")
            ("imageListFile", boost::program_options::value<std::string>(&imageListFile)->default_value("frames_capture_1_1in3_for_psl.txt"), "Image list file")
//            ("configFile", boost::program_options::value<std::string>(&configFile)->default_value("conf3D.txt"), "Config file")
            ("display",  boost::program_options::bool_switch(&display), "Display images and depth maps")
            ("outputDirectory", boost::program_options::value<std::string>(&outDir)->default_value("depthMaps_capture_1_tmp"), "Depth maps output directory")
            ("filter", boost::program_options::bool_switch(&filter),"Filter the depths maps based on cost and uniqueness ratio")
            ("debug", boost::program_options::bool_switch(&debug),"Show more debug informations")
            ("view", boost::program_options::value<int>(&view)->default_value(-1), "View id to generate depthmap for. All if -1.")
            ("PS_MIN_DEPTH", boost::program_options::value<float>(&pSMinDepth)->default_value(0.0), "Min depth")
            ("PS_MAX_DEPTH", boost::program_options::value<float>(&pSMaxDepth)->default_value(0.0), "Min depth")
            ("PS_NUM_PLANES", boost::program_options::value<int>(&pSNumPlanes)->default_value(100), "Number of slices")
            ("PS_RESCALE_FACTOR", boost::program_options::value<float>(&pSRescaleFactor)->default_value(1.0), "Rescale factor")
            ("PS_MATCHING_COSTS", boost::program_options::value<std::string>(&pSMatchingCostsStr)->default_value("SAD"), "Type of matching cost strategy [ZNCC or SAD]")
            ("PS_MATCHING_WINDOW_SIZE", boost::program_options::value<int>(&pSMatchingWindowSize)->default_value(15), "Matching window size")
            ("PS_COLOR_MATCHING", boost::program_options::bool_switch(&pSColorMatching),"Toggle color matching")
            ("krtdOnly", boost::program_options::bool_switch(&krtdOnly),"Don't compute depthmap, juste compute scaled krtd")
            ("PS_AUTO_RANGE", boost::program_options::bool_switch(&pSAutoRange),"Toggle auto range")
            ("PS_OCCLUSION_MODE", boost::program_options::value<std::string>(&pSOcclusionModeStr)->default_value("None"), "Type of occlusion mode [None, BestK or RefSplit]")
            ("PS_OCCLUSION_BEST_K_K", boost::program_options::value<int>(&pSOcclusionBestKK)->default_value(0), "Best K")
            ("PS_USE_SUBPIXEL", boost::program_options::bool_switch(&psEnableSubPixel),"Toggle sub pixel computation")
            ("PS_MAX_NUM_IMAGES", boost::program_options::value<int>(&pSMaxNumImages)->default_value(40), "Max number of images to use for plane sweep")
            ("PS_COST_THRESHOLD", boost::program_options::value<float>(&costThreshold)->default_value(0.0), "Cost threshold. Only need if \"filter\" is toggled")
            ("PS_UNIQUENESS_RATIO_THRESHOLD", boost::program_options::value<float>(&thresholdUniq)->default_value(0.5), "Uniqueness ratio threshold. Only need if \"filter\" is toggled")
            ("PS_MIN_ANGLE_DEGREE", boost::program_options::value<float>(&minAngleDegree)->default_value(2.0), "Min angle degree between two frames")
            ("ref_view_step", boost::program_options::value<int>(&refViewStep)->default_value(0), "Best K")
            ("writePointClouds", boost::program_options::value<std::string>(&writePointClouds)->default_value(""), "Saving data to a file [vrml, vti, vtp, vts or vtpvts]")

//            ("writePointClouds", boost::program_options::bool_switch(&writePointClouds), "Write point clouds in vrml format")
           // ("storeBestCostsAndUniquenessRatios", boost::program_options::bool_switch(&storeBestCostsAndUniquenessRatios), "Store the best costs and the uniqueness ratios in dat files")
            ;

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).run(), vm);
    boost::program_options::notify(vm);

    //Read the .ply File
    std::vector<Eigen::Vector3d> points;

    std::ifstream landmarksFile(landmarksPLY.c_str());
    std::string header;

    landmarksFile >> header;
    while(header != "end_header")
    {
      landmarksFile >> header;
    }

    double x, y, z;
    int id;

    while(landmarksFile >> x >> y >> z >> id)
    {
      points.push_back(Eigen::Vector3d(x,y,z));
    }

//    //For debug purpose
//    for (int i = 0; i < points.size(); ++i) {
//      std::cout << points[i] << std::endl;
//    }

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }

    boost::timer timer,globalTimer;

    PSL::PlaneSweepMatchingCosts pSMatchingCosts;
    if (pSMatchingCostsStr == "ZNCC")
    {
        pSMatchingCosts = PSL::PLANE_SWEEP_ZNCC;
    }
    else if (pSMatchingCostsStr == "SAD")
    {
        pSMatchingCosts = PSL::PLANE_SWEEP_SAD;
    }
    else
    {
        PSL_THROW_EXCEPTION("For parameter PS_MATCHING_COSTS only ZNCC or SAD is allowed.")
    }
    PSL::PlaneSweepOcclusionMode pSOcclusionMode;
    if (pSOcclusionModeStr == "None")
    {
        pSOcclusionMode = PSL::PLANE_SWEEP_OCCLUSION_NONE;
    }
    else if (pSOcclusionModeStr == "BestK")
    {
        pSOcclusionMode = PSL::PLANE_SWEEP_OCCLUSION_BEST_K;
    }
    else if (pSOcclusionModeStr == "RefSplit")
    {
        pSOcclusionMode = PSL::PLANE_SWEEP_OCCLUSION_REF_SPLIT;
    }
    else
    {
        PSL_THROW_EXCEPTION("For parameter PS_MATCHING_COSTS only None or BestK or RefSplit is allowed.")
    }
    std::cout << "Matching mode is "<< (pSOcclusionMode==PSL::PLANE_SWEEP_OCCLUSION_NONE ? "None":(pSOcclusionMode==PSL::PLANE_SWEEP_OCCLUSION_BEST_K ? "BestK":"RefSplit"))<<"..." << std::endl;

    float minAngleCosine = cos(minAngleDegree*M_PI/180.0);

    // First we load the image filenames
    std::ifstream imagesStream;
    imagesStream.open(imageListFile.c_str());
    if (!imagesStream.is_open())
    {
        PSL_THROW_EXCEPTION(std::string("Could not load images list file : "+imageListFile).c_str())
    }
    std::vector<std::string> imageFileNames;
    {
        std::string imageFileName;
        while (imagesStream >> imageFileName)
        {
            boost::filesystem::path p(imageFileName);
            imageFileNames.push_back(p.filename().c_str());
        }
    }
    int numCameras = imageFileNames.size();
    std::cout << numCameras << " filenames found in "<< imageListFile << std::endl;

    // Load the poses file
    // Poses are computed with MAP-Tk
    std::map<int, PSL::CameraMatrix<double> > cameras;
    for (int c = 0; c < numCameras; c++)
    {
        int id = c;
        // get filename
        std::string krtdMatrixFileName = imageFileNames[c];
        {
            std::string krtdExt = "krtd";
            size_t start_pos = krtdMatrixFileName.find(".");
            if(start_pos == std::string::npos)
                 PSL_THROW_EXCEPTION(("Could not generate krt filename, no '.' found in "+krtdMatrixFileName).c_str())
            krtdMatrixFileName.replace(start_pos + 1, krtdMatrixFileName.length()-start_pos, krtdExt);
        }
        std::string krtdMatrixFile = krtdFolder + "/" + krtdMatrixFileName;
        // try to load k, r, t matrices
        std::ifstream krtdMatrixStr;
        krtdMatrixStr.open(krtdMatrixFile.c_str());
//        if (!krtdMatrixStr.is_open())
//        {
//            PSL_THROW_EXCEPTION(("Error opening KRT matrix file : "+imageListFile).c_str())
//        }

        if (krtdMatrixStr.is_open())
        {
          Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
          krtdMatrixStr >> K(0,0) >> K(0,1) >> K(0,2);
          krtdMatrixStr >> K(1,0) >> K(1,1) >> K(1,2);
          krtdMatrixStr >> K(2,0) >> K(2,1) >> K(2,2);
          //std::cout << K << std::endl;

          Eigen::Matrix<double, 3, 3> R;
          Eigen::Matrix<double, 3, 1> T;

          for (int i = 0; i < 3; i++)
          {
              for (int j = 0; j < 3; j++)
                  krtdMatrixStr >> R(i,j);

          }
          for (int i = 0; i < 3; i++)
              krtdMatrixStr >> T(i);

          //std::cout << R << std::endl;
          //std::cout << T << std::endl;

          Eigen::Matrix<double, 3, 1> d;
          for (int i = 0; i < 3; i++)
              krtdMatrixStr >> d(i);

          cameras[id].setKRT(K, R, T);
        }
        else
        {
          std::cout << "No krtd file found for the frame " << krtdMatrixFile.c_str() << ". Jumping to next frame..." << std::endl;
        }
    }
    std::cout << "Cameras KRT have been loaded." << std::endl;

    // The reconstructions are not metric. In order to have an idea about the scale
    // we compute the average distance between the cameras.
    double avgDistance = 0, maxDistance=0;
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
                    double d = distance.norm();
                    avgDistance += d;
                    maxDistance = std::max(maxDistance,d);
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
    std::cout << "Cameras have an average distance of " << avgDistance << " and a max distance of " << maxDistance <<  "." << std::endl;
    float minZ = (float) (2.5f*avgDistance);
    float maxZ = (float) (100.0f*avgDistance);

    int imageWidth=1920;
    int imageHeight=1080;
    std::cout<< "Images are assumed to be " << imageWidth << "x"<< imageHeight <<std::endl;
    makeOutputFolder(outDir);

    globalTimer.restart();
    int count = 0;
    std::string listName;
    std::ofstream filenameListvts, filenameListvtp, filenameListvti;
    if (!krtdOnly){
      listName = outDir + "/vtsList.txt";
      filenameListvts.open(listName.c_str());
      listName = outDir + "/vtpList.txt";
      filenameListvtp.open(listName.c_str());
      listName = outDir + "/vtiList.txt";
      filenameListvti.open(listName.c_str());
    }
    listName = outDir + "/kList.txt";
    std::ofstream kList(listName.c_str());
    for (std::map<int, PSL::CameraMatrix<double> >::iterator it = cameras.begin(); it != cameras.end(); std::advance(it, refViewStep), count++)
    {
        // if we want a specific ref view, go for it.
        if(view>=0 && it->first!=view) continue;
        const int refViewId = it->first;
        timer.restart();
        std::cout << "Find overlaps for ref cam: " << refViewId << "..." << std::endl;
        std::vector<std::pair<int, float> > overlaps;
        std::cout << "WARNING Overlap prior is small angle for now." << std::endl;
        for (int i = 0; i < cameras.size(); ++i) {
            if(i!=it->first) overlaps.push_back(std::make_pair(i,
                                                               cameraDirAngleCosine(cameras[refViewId],cameras[i])
                                                               ));
        }
        std::sort(overlaps.begin(), overlaps.end(), overlapCompare);

        std::cout << " done in " << timer.elapsed() << " seconds." << std::endl;
        std::cout << " Best overlapping cams in descending order: ";
        for (unsigned int j = 0; j < overlaps.size(); j++)
        {
            std::cout << overlaps[j].first << " ";
        }
        std::cout << std::endl;


        timer.restart();
        std::cout << "Setting up cuda plane sweep and upload images..." << std::endl;
        PSL::CudaPlaneSweep cPS;
        cPS.setScale(pSRescaleFactor);
        if (debug) std::cout << "  pSRescaleFactor " << pSRescaleFactor <<  std::endl;
        cPS.setMatchWindowSize(pSMatchingWindowSize,pSMatchingWindowSize);
        if (debug) std::cout << "  pSMatchingWindowSize " << pSMatchingWindowSize <<  std::endl;
        cPS.setNumPlanes(pSNumPlanes);
        if (debug) std::cout << "  pSNumPlanes " << pSNumPlanes <<  std::endl;
        cPS.setOcclusionMode(pSOcclusionMode);
        if (debug) std::cout << "  pSOcclusionMode " << pSOcclusionMode <<  std::endl;
        if (pSOcclusionMode == PSL::PLANE_SWEEP_OCCLUSION_BEST_K){
            cPS.setOcclusionBestK(pSOcclusionBestKK);
            if (debug) std::cout << "  pSOcclusionBestKK " << pSOcclusionBestKK <<  std::endl;
        }
        cPS.setMatchingCosts(pSMatchingCosts);
        if (debug) std::cout << "  pSMatchingCosts " << pSMatchingCosts <<  std::endl;
        cPS.setPlaneGenerationMode(PSL::PLANE_SWEEP_PLANEMODE_UNIFORM_DISPARITY);
        cPS.setSubPixelInterpolationMode(PSL::PLANE_SWEEP_SUB_PIXEL_INTERP_INVERSE);
        cPS.enableSubPixel(psEnableSubPixel);
        cPS.enableColorMatching(pSColorMatching);
        if (debug) std::cout << "  pSColorMatching " << pSColorMatching <<  std::endl;

        if (pSAutoRange)
        {


            // Compute near and far range.
            double nearZ,farZ;
            std::vector<float> zs;
            for(unsigned int i=0; i<points.size(); i++) {
                Eigen::Vector3d x = (it->second.getR()*points[i] + it->second.getT());
                Eigen::Vector3d px= 1.0/x[2]*(it->second.getK()*x);
                if(x[2]>0 && px[0]>0 && px[0]<imageWidth && px[1]>0 && px[1]<imageHeight)
                    zs.push_back(x[2]);
            }
            sort(zs.begin(),zs.end());
            const double outlierThres=0.05;
            double safeness_margin_factor=0.33;
            if (zs.size()){
                nearZ = zs[(int)((zs.size()-1)*outlierThres)];
                farZ = zs[(int)((zs.size()-1)*(1-outlierThres))];
                std::cout<<"    distances from camera are "<<zs[0]<<"->"<<zs[zs.size()-1]<<" ("<< zs.size()<<" visible points)"<<std::endl;
                std::cout<<"    distances ["<<outlierThres*100<<"%->"<<(1.0-outlierThres)*100<<"%] are "<<nearZ<<"->"<<farZ<<std::endl;
                nearZ *= (1-safeness_margin_factor);
                farZ *= (1+safeness_margin_factor);
                std::cout<<"    planes will be "<<nearZ<<"->"<<farZ<<" by "<<(farZ-nearZ)/pSNumPlanes<<" increment"<<std::endl;
                pSMinDepth=nearZ;
                pSMaxDepth=farZ;
                cPS.setZRange(nearZ, farZ);
            }else {
                std::cout<<"    NO VISIBLE POINTS, demoting to depths from config file : "<<pSMinDepth<<"->"<< pSMaxDepth<<" ."<<std::endl;

            }
            maxZ  = pSMaxDepth;
            minZ  = pSMinDepth;
        }
        else {
          maxZ  = pSMaxDepth;
          minZ  = pSMinDepth;
          cPS.setZRange(minZ, maxZ);
        }

        if (debug) std::cout << "  Z range :  " << pSMinDepth << "  - " << pSMaxDepth <<  std::endl;

        if ((filter || storeBestCostsAndUniquenessRatios)/* && !sgm*/)
        {
            cPS.enableOutputBestCosts();
            cPS.enableOuputUniquenessRatio();
        }

        if (!((unsigned int) it->first < imageFileNames.size()))
        {
            PSL_THROW_EXCEPTION("Reference camera image not in image list")
        }
        std::string refImgFileName = frameFolder + "/" + imageFileNames[it->first];
        std::string baseName = PSL::extractBaseFileName(refImgFileName);
        cv::Mat refImg = cv::imread(refImgFileName);
        if (refImg.empty())
            PSL_THROW_EXCEPTION(("Failed to load image : "+refImgFileName).c_str())

        //First write K File
        {
          PSL::CameraMatrix<double> refCamScaled = cameras[refViewId];
          refCamScaled.scaleK(pSRescaleFactor,pSRescaleFactor);

          std::ostringstream kFileName;
          kFileName << outDir << "/" << baseName << "_camera.krtd";
          std::ofstream kFile;
          kFile.open(kFileName.str().c_str());
          kFile << refCamScaled.getK();
          kFile << std::endl << std::endl;
          kFile << refCamScaled.getR();
          kFile << std::endl << std::endl;
          kFile << refCamScaled.getT()(0,0) << " " << refCamScaled.getT()(1,0) << " " << refCamScaled.getT()(2,0) << std::endl;
          kFile << std::endl << std::endl;
          kFile << "0";
          kFile.close();
          kList << refViewId << " " <<  baseName << "_camera.krtd" << std::endl;
          std::cout << "Saved : " << kFileName.str() << std::endl;
          if (krtdOnly) continue;
        }

        PSL::CameraMatrix<double>const & refCam = it->second;
        int numMatchingImages = 0;
        std::vector<int> selectedImg;
        for (int j = 0; j < overlaps.size(); j++)
        {
            if ( numMatchingImages >= pSMaxNumImages )
                continue;
            const int viewJ=overlaps[j].first;
            double cosine = cameraDirAngleCosine(refCam,cameras[viewJ]);
            if(cosine < minAngleCosine)
            {
                selectedImg.push_back(viewJ);
                numMatchingImages++;
            }
        }

        selectedImg.push_back(it->first);
        int cPSRef;
        std::sort(selectedImg.begin(), selectedImg.end());
        for (int j = 0; j < selectedImg.size(); j++){
            const int viewJ=selectedImg[j];
            std::string imageFileName = frameFolder + "/" + imageFileNames[viewJ];

            cv::Mat img = cv::imread(imageFileName);

            int cudaID=cPS.addImage(img, cameras[viewJ]);
            if(viewJ==it->first)
                cPSRef=cudaID;
            if (debug) std::cout << "Loaded image : " << imageFileName << std::endl;
        }
        std::cout<<" Found "<<numMatchingImages<<" matching images (Angle>"<<180.0/M_PI*acos(minAngleCosine)<<" degrees)" << std::endl;
        std::cout << " done in " << timer.elapsed() << " seconds." << std::endl;

        if (pSOcclusionMode==PSL::PLANE_SWEEP_OCCLUSION_BEST_K && numMatchingImages < pSOcclusionBestKK)
        {
            std::cout << "WARNING: skipped reference image " << it->first << " because there where not enough matching images for given best K, K." << std::endl;
            continue;
        }
        if (pSOcclusionMode==PSL::PLANE_SWEEP_OCCLUSION_REF_SPLIT){
            std::vector<int>::iterator p = std::find (selectedImg.begin(), selectedImg.end(), it->first);
            int numBefore = p-selectedImg.begin();
            int numAfter = selectedImg.size() - numBefore;
            if( std::min(numBefore,numAfter)<(numMatchingImages/4))
                cPS.setOcclusionMode(PSL::PLANE_SWEEP_OCCLUSION_NONE);
        }
        timer.restart();

        std::cout << "Running cuda plane sweep... cPSRef = " << cPSRef << std::endl;
        cPS.process(cPSRef);
        std::cout << " done in " << timer.elapsed() << " seconds." << std::endl;

        PSL::DepthMap<float, double> dM;
        {
            std::cout << "Download best depths..." << std::endl;
            dM=cPS.getBestDepth();
            std::cout << "done." << std::endl;
            if (filter || storeBestCostsAndUniquenessRatios)
            {
                std::cout << "Download best costs and uniqueness ratios..." << std::endl;
                PSL::Grid<float> costs = cPS.getBestCosts();
                PSL::Grid<float> uniquenessRatios = cPS.getUniquenessRatios();

                std::cout << "done." << std::endl;

                if (storeBestCostsAndUniquenessRatios)
                {
                    std::ostringstream uniqunessFileName;
                    std::ostringstream bestCostFileName;
                    std::ostringstream bestCostImgFileName;

                    uniqunessFileName << outDir << "/" << baseName << "_uniquenessRatios.dat";
                    uniquenessRatios.saveAsDataFile(uniqunessFileName.str());

                    float maxBestCost = (pSMatchingCosts==PSL::PLANE_SWEEP_SAD)?(pSMatchingWindowSize*pSMatchingWindowSize)*255.0f:1.0f;
                    bestCostImgFileName << outDir << "/" << baseName << "_bestCosts.jpg";
                    std::cout << "maxBestCost = " << maxBestCost << std::endl;
                    PSL::saveGridZSliceAsImage(costs, 0, 0.0f, maxBestCost, bestCostImgFileName.str().c_str());

                    bestCostFileName << outDir << "/" << baseName << "_bestCosts.dat";
                    costs.saveAsDataFile(bestCostFileName.str());

                    std::ostringstream uniqunessImgFileName;
                    uniqunessImgFileName << outDir << "/" << baseName << "_uniquenessRatios.jpg";
                    PSL::saveGridZSliceAsImage(uniquenessRatios, 0, 0.8f, 1.0f, uniqunessImgFileName.str().c_str());
                }

                if (filter)
                {


                    int w = dM.getWidth(); int h = dM.getHeight();
                    for (int y = 0; y < h; y++)
                    {
                        for(int x = 0; x<w; ++x)
                        {
                          std::cout << "costs(x,y) = " << costs(x,y) <<std::endl;
                            if(costs(x,y)>costThreshold ||
                               uniquenessRatios(x,y) > thresholdUniq)
                                dM(x,y)=-1;
                        }
                    }
                }
            }
        }


//        std::ostringstream fileNameData;
//        fileNameData << outDir << "/" << baseName << "_depth_map.dat";
//        dM.saveAsDataFile(fileNameData.str());
//        std::cout << "Saved : " << fileNameData.str() << std::endl;
        std::ostringstream fileNameImg;
        fileNameImg << outDir << "/" << baseName << "_inv_depth_map.jpg";
        dM.saveInvDepthAsColorImage(fileNameImg.str(), minZ, maxZ);
        std::cout << "Saved : " << fileNameImg.str() << std::endl;


//        char kNameAbs[PATH_MAX];
//                realpath(kFileName.str().c_str(), kNameAbs);
        if (writePointClouds != "")
        {
            // scale refImg

            cv::Mat scaledRefImg;

            if (pSRescaleFactor != 1)
            {
                // scale image
                if (pSRescaleFactor < 1)
                    resize(refImg, scaledRefImg, cv::Size(0,0), pSRescaleFactor, pSRescaleFactor, cv::INTER_AREA);
                else
                    resize(refImg, scaledRefImg, cv::Size(0,0) , pSRescaleFactor, pSRescaleFactor, cv::INTER_LINEAR);
            }
            else
            {
                scaledRefImg = refImg.clone();
            }

            //Convert to VRML file
            if(writePointClouds == "vrml") {
              std::ostringstream pointCloudFileName;
              pointCloudFileName << outDir << "/" << baseName << "_point_cloud.wrl";
              std::ofstream pointCloudFile;
              pointCloudFile.open(pointCloudFileName.str().c_str());
              if (!pointCloudFile.is_open())
              {
                  PSL_THROW_EXCEPTION("Error opening VRML file for writing");
              }
              dM.pointCloudColoredToVRML(pointCloudFile,scaledRefImg);
              std::cout << "Saved : " << pointCloudFileName.str() << std::endl;
            }
            //Convert to VTP and/or VTS file
            else
            {
              vtkNew<vtkPoints> points;
              points->SetNumberOfPoints(scaledRefImg.rows*scaledRefImg.cols);

              vtkNew<vtkDoubleArray> uniquenessRatios;
              uniquenessRatios->SetName("Uniqueness Ratios");
              uniquenessRatios->SetNumberOfValues(scaledRefImg.rows*scaledRefImg.cols);

              vtkNew<vtkDoubleArray> bestCost;
              bestCost->SetName("Best Cost Values");
              bestCost->SetNumberOfValues(scaledRefImg.rows*scaledRefImg.cols);

              vtkNew<vtkUnsignedCharArray> color;
              color->SetName("Color");
              color->SetNumberOfComponents(3);
              color->SetNumberOfTuples(scaledRefImg.rows*scaledRefImg.cols);

              vtkNew<vtkDoubleArray> depths;
              depths->SetName("Depths");
              depths->SetNumberOfComponents(1);
              depths->SetNumberOfTuples(scaledRefImg.rows*scaledRefImg.cols);

              PSL::Grid<float> costs = cPS.getBestCosts();
              PSL::Grid<float> uRatios = cPS.getUniquenessRatios();

              int dim[3] = {dM.getWidth(),dM.getHeight(),1};
              int ijk[3];
              ijk[2] = 0;

              vtkIdType pt_id;

              for (int x = 0; x < dM.getWidth(); ++x) {
                ijk[0] = x;
                for (int y = 0; y < dM.getHeight(); ++y) {
                  ijk[1] = dM.getHeight()-y-1;
                  pt_id = vtkStructuredData::ComputePointId(dim,ijk);

                  Eigen::Matrix<double, 4, 1> p = dM.unproject(x,y);

                  points->SetPoint(pt_id,p(0,0),p(1,0),p(2,0));
                  uniquenessRatios->SetValue(pt_id, uRatios(x,y));
                  bestCost->SetValue(pt_id, costs(x,y));

                  depths->SetValue(pt_id,dM(x,y));

                  cv::Vec3b bgr = scaledRefImg.at<cv::Vec3b>(y,x);
                  color->SetTuple3(pt_id,(int) bgr[2],(int) bgr[1],(int) bgr[0]);
                }
              }

              vtkNew<vtkImageData> imageData;
              imageData->SetSpacing(1,1,1);
              imageData->SetOrigin(0,0,0);
              imageData->SetDimensions(dM.getWidth(),dM.getHeight(),1);
              imageData->GetPointData()->AddArray(depths.Get());
              imageData->GetPointData()->AddArray(color.Get());
              imageData->GetPointData()->AddArray(uniquenessRatios.Get());
              imageData->GetPointData()->AddArray(bestCost.Get());

              vtkNew<vtkXMLImageDataWriter> writerI;
              std::string refFrame = static_cast<ostringstream*>( &(ostringstream() << refViewId/refViewStep) )->str();
              std::string depthmapImageFileName = outDir + "/"+ baseName + "_depth_map." + refFrame + ".vti";

              writerI->SetFileName(depthmapImageFileName.c_str());
              writerI->AddInputDataObject(imageData.Get());
//              writerI->SetCompressorTypeToZLib();
              writerI->SetDataModeToBinary();
              writerI->Write();
              std::cout << "Saved : " << depthmapImageFileName << std::endl;

//              char depthmapImageNameAbs[PATH_MAX];
//                      realpath(depthmapImageFileName.c_str(), depthmapImageNameAbs);
              filenameListvti << refViewId << " " << baseName << "_depth_map." << refFrame << ".vti" << std::endl;

              //Writing polydata to the disk
              if (writePointClouds == "vtp" || writePointClouds == "vtpvts") {

                vtkNew<vtkPolyData> polydata;
                polydata->SetPoints(points.Get());
                polydata->GetPointData()->AddArray(depths.Get());
                polydata->GetPointData()->AddArray(uniquenessRatios.Get());
                polydata->GetPointData()->AddArray(bestCost.Get());
                polydata->GetPointData()->AddArray(color.Get());

                vtkNew<vtkXMLPolyDataWriter> writerP;
                std::string depthmapPolyFileName = outDir + "/"+ baseName + "_depth_map." + refFrame + ".vtp";

                writerP->SetFileName(depthmapPolyFileName.c_str());
                writerP->AddInputDataObject(polydata.Get());
//                writerP->SetCompressorTypeToZLib();
                writerP->SetDataModeToBinary();
                writerP->Write();
                std::cout << "Saved : " << depthmapPolyFileName << std::endl;

//                char depthmapPolyFileNameAbs[PATH_MAX];
//                        realpath(depthmapPolyFileName.c_str(), depthmapPolyFileNameAbs);
                filenameListvtp << refViewId << " " << depthmapPolyFileName.c_str() << std::endl;
              }

              //Writing structured grid to the disk
              if (writePointClouds == "vts" || writePointClouds == "vtpvts") {


                vtkNew<vtkStructuredGrid> structuredGrid;
                structuredGrid->SetDimensions(dM.getWidth(),dM.getHeight(),1);
                structuredGrid->SetPoints(points.Get());
                structuredGrid->GetPointData()->AddArray(depths.Get());
                structuredGrid->GetPointData()->AddArray(uniquenessRatios.Get());
                structuredGrid->GetPointData()->AddArray(bestCost.Get());
                structuredGrid->GetPointData()->AddArray(color.Get());

                vtkNew<vtkXMLStructuredGridWriter> writerG;
//                std::string refFrame = static_cast<ostringstream*>( &(ostringstream() << refViewId/refViewStep) )->str();
                std::string depthmapGridFileName = outDir + "/"+ baseName + "_depth_map." + refFrame + ".vts";

                writerG->SetFileName(depthmapGridFileName.c_str());
                writerG->AddInputDataObject(structuredGrid.Get());
//                writerG->SetCompressorTypeToZLib();
                writerG->SetDataModeToBinary();
                writerG->Write();
                std::cout << "Saved : " << depthmapGridFileName << std::endl;

//                char depthmapGridFileNameAbs[PATH_MAX];
//                        realpath(depthmapGridFileName.c_str(), depthmapGridFileNameAbs);
                filenameListvts << refViewId << " " << depthmapGridFileName.c_str() << std::endl;


              }
            }
        }

        if (display)
            dM.displayInvDepthColored(pSMinDepth, pSMaxDepth, 20);

        std::cout << std::endl;
        int done=count+1,tot=cameras.size()/refViewStep,tbd=tot-done;
        std::cout << "Elapsed time: "<< globalTimer.elapsed()<< "s for " << done << "/" << tot << " ref views."<<std::endl
                  << "Estimated Time Remaining: " << globalTimer.elapsed()/done*tbd << "s for "
                  << tbd << "/" << tot << " ref views."<<std::endl;
    }
    filenameListvts.close();
    filenameListvtp.close();
    filenameListvti.close();
    kList.close();
}
