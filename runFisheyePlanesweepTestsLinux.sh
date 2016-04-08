#!/bin/sh  

if [ -d "testResults/fisheyeCamera/" ]; then
  echo "Old results found. They need to be deleted before running the tests again, do you want to delete them?"
  rm -r -I testResults/fisheyeCamera/
fi

if [ ! -d "testResults/fisheyeCamera/" ]; then
  mkdir -p testResults/fisheyeCamera/

  echo "Runing fisheye planesweep on the left dataset..."
  build/bin/fisheyePlanesweepTest --dataFolder ./data/fisheyeCamera/left/
  mv fisheyeTestResults testResults/fisheyeCamera/left/
  echo "done"

  echo "Runing fisheye planesweep on the right dataset..."
  build/bin/fisheyePlanesweepTest --dataFolder ./data/fisheyeCamera/right/
  mv fisheyeTestResults testResults/fisheyeCamera/right/
  echo "done" 
fi

