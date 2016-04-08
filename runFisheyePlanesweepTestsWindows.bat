@echo off

if exist .\testResults\fisheyeCamera (
	echo Old results found. They need to be deleted before running the tests again, do you want to delete them?
	rmdir /S .\testResults\fisheyeCamera
	)
	
if not exist .\testResults\fisheyeCamera (
	mkdir .\testResults\fisheyeCamera

	echo Runing fisheye planesweep on the left dataset...
	build\bin\Release\fisheyePlanesweepTest --dataFolder .\data\fisheyeCamera\left\
	move fisheyeTestResults testResults\fisheyeCamera\left
	echo done

	echo Runing fisheye planesweep on the right dataset...
	build\bin\Release\fisheyePlanesweepTest --dataFolder .\data\fisheyeCamera\right\
	move fisheyeTestResults testResults\fisheyeCamera\right
	echo done
)


	