@echo off
rem ---------------------------------------------------------------------------------------
rem Build RtgmDif (x64 Release).
rem Output: RtgmDif\build\RtgmDif.dll + RtgmDif.lib + nnedi3_weights.bin
rem
rem This project and RtgmDifTest are self-contained: everything they compile is either under
rem RtgmDif\ or RtgmDifTest\, so the two directories can be moved out of the NVEnc tree as a
rem pair. The only things they need from outside are the CUDA toolkit and the Windows SDK.
rem
rem  /m:2 is intentional: the RTGMC kernels are large and running too many nvcc/cicc
rem  instances in parallel makes cicc die with "LLVM error : out of memory".
rem ---------------------------------------------------------------------------------------
setlocal

set "MSBUILD=C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\MSBuild\Current\Bin\MSBuild.exe"
if not exist "%MSBUILD%" (
  for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe`) do set "MSBUILD=%%i"
)

"%MSBUILD%" "%~dp0RtgmDif.vcxproj" /p:Configuration=Release /p:Platform=x64 /m:2 /nologo /v:m %*
exit /b %ERRORLEVEL%
