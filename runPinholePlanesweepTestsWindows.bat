@echo off

if exist .\testResults\pinholeCamera (
	echo Old results found. They need to be deleted before running the tests again, do you want to delete them?
	rmdir /S .\testResults\pinholeCamera
	)
	
if not exist .\testResults\pinholeCamera (
	mkdir .\testResults\pinholeCamera

	echo Runing pinhole planesweep on the niederdorf1 dataset...
	build\bin\Release\pinholePlanesweepTest --dataFolder .\data\pinholeCamera\niederdorf1\
	move pinholeTestResults testResults\pinholeCamera\niederdorf1
	echo done

	echo Runing pinhole planesweep on the niederdorf2 dataset...
	build\bin\Release\pinholePlanesweepTest --dataFolder .\data\pinholeCamera\niederdorf2\
	move pinholeTestResults testResults\pinholeCamera\niederdorf2
	echo done

	echo Runing pinhole planesweep on the niederdorf3 dataset...
	build\bin\Release\pinholePlanesweepTest --dataFolder .\data\pinholeCamera\niederdorf3\
	move pinholeTestResults testResults\pinholeCamera\niederdorf3
	echo done
)
	