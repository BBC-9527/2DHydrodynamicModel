@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat" -arch=x64
cl /O2 /LD /EHsc /Fe:swe_coupling.dll src/swe_coupling.cpp
if %errorlevel% neq 0 echo Compilation Failed & exit /b 1
echo Compilation Success
del swe_coupling.obj swe_coupling.exp swe_coupling.lib
