@echo off
rem ---------------------------------------------------------------------------------------
rem Build RtgmDifTest (x64 Release).
rem Output: RtgmDifTest\build\RtgmDifTest.exe (+ RtgmDif.dll and nnedi3_weights.bin, copied in)
rem
rem RtgmDif.dll has to be built first (..\RtgmDif\build.bat). This project only links the
rem import library and cudart, and reaches outside its own directory for nothing but the
rem sibling RtgmDif project - so the two directories can be moved out of the NVEnc tree as a
rem pair.
rem ---------------------------------------------------------------------------------------
setlocal

set "MSBUILD=C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\MSBuild\Current\Bin\MSBuild.exe"
if not exist "%MSBUILD%" (
  for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe`) do set "MSBUILD=%%i"
)

"%MSBUILD%" "%~dp0RtgmDifTest.vcxproj" /p:Configuration=Release /p:Platform=x64 /m /nologo /v:m
exit /b %ERRORLEVEL%
