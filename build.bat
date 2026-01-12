@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat" -arch=x64
cl /O2 /LD /EHsc /Fe:swe_core.dll src/swe_core.cpp
if %errorlevel% neq 0 echo Compilation Failed (swe_core) & exit /b 1

cl /O2 /LD /EHsc /Fe:swe_1d_core.dll src/swe_1d_core.cpp
if %errorlevel% neq 0 echo Compilation Failed (swe_1d_core) & exit /b 1

cl /O2 /LD /EHsc /Fe:swe_coupling.dll src/swe_coupling.cpp
if %errorlevel% neq 0 echo Compilation Failed (swe_coupling) & exit /b 1

echo Compilation Success
del swe_core.obj swe_core.exp swe_core.lib
del swe_1d_core.obj swe_1d_core.exp swe_1d_core.lib
del swe_coupling.obj swe_coupling.exp swe_coupling.lib
