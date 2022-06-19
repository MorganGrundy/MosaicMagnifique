@echo off
setlocal

REM Set definitions
SET main_dir=%~dp0
IF NOT DEFINED opencvversion SET opencvversion=4.3.0
SET opencvzip=%opencvversion%.zip
IF NOT DEFINED opencv SET opencv=C:\OpenCV

cd /d "%main_dir%"

IF DEFINED mode (
	IF "%mode%" == "opencv" (
		Call :GetOpenCV
		Call :BuildOpenCV
		Call :OpenCVEnvironmentVariable
		exit /b
	) ELSE (
		IF "%mode%" == "environment" (
			echo "Environment"
			Call :OpenCVEnvironmentVariable
			exit /b
		) ELSE (
			IF "%mode%" == "all" (
				Call :GetOpenCV
				Call :BuildOpenCV
				Call :OpenCVEnvironmentVariable
				exit /b
			)
		)
	)
)
Call :HelpInfo
exit /b

REM Download and unzip OpenCV source
:GetOpenCV
IF NOT DEFINED wgetdir (
	echo wget not found
	exit /b
)
IF NOT EXIST "%wgetdir%\wget" (
	echo wget not found at "%wgetdir%\wget", ensure %%wgetdir%% is the path to wget executable.
	exit /b
)
mkdir "%opencv%"
echo "%wgetdir%\wget" https://github.com/opencv/opencv/archive/%opencvzip%
Call :UnZipFile "%opencv%" "%main_dir%%opencvzip%"
exit /b

REM Builds OpenCV release from source
:BuildOpenCV
IF NOT EXIST "%opencv%\opencv-%opencvversion%" (
	echo OpenCV source not found at "%opencv%\opencv-%opencvversion%"
	exit /b
)
cd /d "%opencv%\opencv-%opencvversion%"
mkdir build
cd build
cmake -DBUILD_LIST=calib3d,core,features2d,flann,highgui,imgcodecs,imgproc,objdetect -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config release
cmake --build . --target install --config release
exit /b

REM Adds OpenCV Environment Variables
:OpenCVEnvironmentVariable
REM Finds if OpenCV install is 64-bit or 32-bit
IF EXIST "%opencv%\opencv-%opencvversion%\build\install\x64\" (
	SET WINDOW_BIT=x64
) ELSE (
	SET WINDOW_BIT=x86
)

REM Finds highest Visual C++ version
REM Gets list of directories in opencv install sorted by name alphabetically, gets last entry
for /f "delims=" %%a in ('dir /a:d /b /o:n "%opencv%\opencv-%opencvversion%\build\install\%WINDOW_BIT%"') do SET VC_VERSION=%%a
REM Ensure that build path exists
IF NOT EXIST "%opencv%\opencv-%opencvversion%\build\install\%WINDOW_BIT%\%VC_VERSION%" (
	echo Failed to set environment variable as OpenCV build was not found
	echo Searched at "%opencv%\opencv-%opencvversion%\build\install\%WINDOW_BIT%\%VC_VERSION%"
	exit /b
)
setx /M OPENCV_DIR "%opencv%\opencv-%opencvversion%\build\install\%WINDOW_BIT%\%VC_VERSION%"
setx /M PATH "%PATH%;%%OPENCV_DIR%%\bin"
exit /b

REM Unzips <newzipfile> at <ExtractTo>
:UnZipFile <ExtractTo> <newzipfile>
set vbs="%temp%\_.vbs"
if exist %vbs% del /f /q %vbs%
>%vbs%  echo Set fso = CreateObject("Scripting.FileSystemObject")
>>%vbs% echo If NOT fso.FolderExists(%~1) Then
>>%vbs% echo fso.CreateFolder(%~1)
>>%vbs% echo End If
>>%vbs% echo set objShell = CreateObject("Shell.Application")
>>%vbs% echo set FilesInZip=objShell.NameSpace(%~2).items
>>%vbs% echo objShell.NameSpace(%~1).CopyHere(FilesInZip)
>>%vbs% echo Set fso = Nothing
>>%vbs% echo Set objShell = Nothing
cscript //nologo %vbs%
if exist %vbs% del /f /q %vbs%
exit /b

REM Displays help info
:HelpInfo
echo ----- Mosaic Magnifique requirements install helper -----
echo Downloading OpenCV requires wget. Set variable %%wgetdir%% such that wget can be found at %%wgetdir%%\wget.exe
echo.
echo Set %%mode%% variable to control script actions:
echo - "opencv" will download OpenCV source using wget, build a minimal release using cmake, and set environment variables.
echo - "environment" will create %%OPENCV_DIR%% system variable and add %%OPENCV_DIR%%\bin to system %%PATH%% variable.
echo - "all" will perform all steps.
echo.
echo OpenCV will be installed at "%opencv%". To change this set %%opencv%% variable to the wanted path.
echo OpenCV version to install will be %opencvversion%, to change this set %%opencvversion%% variable to wanted version.
exit /b