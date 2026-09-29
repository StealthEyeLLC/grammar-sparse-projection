@echo off
setlocal
set "GSP_ROOT=%~dp0.."
if not defined GSP_PYTHON set "GSP_PYTHON=python"
if exist "C:\VS2022BuildTools\Common7\Tools\VsDevCmd.bat" call "C:\VS2022BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul
if exist "%GSP_ROOT%\vendor\triton\backends\nvidia\bin\ptxas.exe" (
  set "CUDA_PATH=%GSP_ROOT%\vendor\triton\backends\nvidia"
  set "CUDA_HOME=%GSP_ROOT%\vendor\triton\backends\nvidia"
  set "PATH=%GSP_ROOT%\vendor\triton\backends\nvidia\bin;%PATH%"
)
"%GSP_PYTHON%" %*
exit /b %ERRORLEVEL%
