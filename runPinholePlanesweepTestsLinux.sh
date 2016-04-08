#!/bin/sh  

if [ -d "testResults/pinholeCamera/" ]; then
  echo "Old results found. They need to be deleted before running the tests again, do you want to delete them?"
  rm -r -I testResults/pinholeCamera
fi

if [ ! -d "testResults/pinholeCamera/" ]; then
  mkdir -p testResults/pinholeCamera/

  echo "Runing pinhole planesweep on the niederdorf1 dataset..."
  build/bin/pinholePlanesweepTest --dataFolder ./data/pinholeCamera/niederdorf1/
  mv pinholeTestResults testResults/pinholeCamera/niederdorf1
  echo "done" 

  echo "Runing pinhole planesweep on the niederdorf2 dataset..."
  build/bin/pinholePlanesweepTest --dataFolder ./data/pinholeCamera/niederdorf2/
  mv pinholeTestResults testResults/pinholeCamera/niederdorf2
  echo "done" 

  echo "Runing pinhole planesweep on the niederdorf3 dataset..."
  build/bin/pinholePlanesweepTest --dataFolder ./data/pinholeCamera/niederdorf3/
  mv pinholeTestResults testResults/pinholeCamera/niederdorf3
  echo "done" 
fi
