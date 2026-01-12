@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat" -arch=x64
cl /O2 /LD /EHsc /Fe:swe_1d_network.dll src/swe_1d_network.cpp
if %errorlevel% neq 0 echo Compilation Failed & exit /b 1
echo Compilation Success
del swe_1d_network.obj swe_1d_network.exp swe_1d_network.lib
