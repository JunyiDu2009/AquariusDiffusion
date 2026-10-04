@echo off
REM ---------------------------------------------------------------------------
REM AquariusTerimageDemo -- text2img web UI launcher (Windows)
REM
REM Double-click this file. It runs app.py in this folder.
REM
REM NOTE: this .bat is deliberately ASCII-only with CRLF line endings.
REM   cmd.exe on a GBK(936) console decodes UTF-8 non-ASCII bytes as GBK; a
REM   multi-byte char whose tail byte is 0x81-0xFE is read as a GBK lead byte
REM   and SWALLOWS the following newline, gluing lines together and breaking
REM   the script. The symptom is: double-clicked and nothing happened.
REM   Do NOT put non-ASCII characters in this file. Keep CRLF.
REM   The same rule applies to every file in this package: all file names,
REM   folder names and text are ASCII on purpose.
REM
REM Args pass through, e.g.:
REM   run_ui.bat --port 8888
REM   run_ui.bat --lowvram
REM   run_ui.bat --device cpu
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

echo ===========================================================
echo   AquariusTerimageDemo  -  text2img web UI
echo -----------------------------------------------------------
echo   Model: a ternary UNet at training step 260000.
echo   Output is still blurry scene texture WITHOUT recognizable
echo   objects. That is EXPECTED of this checkpoint, not an
echo   install problem.
echo -----------------------------------------------------------
echo   Startup takes 30-90s (loading torch + 3 models).
echo   A quiet window is NORMAL. After that each image takes
echo   a few seconds.
echo ===========================================================
echo.

REM --- find a python that actually has the deps (never trust PATH) ----------
set PY=
call :try "%~dp0.venv\Scripts\python.exe"
call :try "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
call :try "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
call :try "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
call :try "C:\Python313\python.exe"
call :try "C:\Python312\python.exe"
call :try "python"
if "%PY%"=="" (
  echo ERROR: no usable python found. Install Python 3.11-3.13 and the
  echo        requirements, then run this file again.
  echo        See requirements.txt for the exact pip commands.
  pause
  exit /b 1
)
echo Using: %PY%
echo.

REM Text-encoder residency: app.py already defaults to --te-mode resident, i.e.
REM the TE weights stay pinned in VRAM.  Measured on a 32GB card at 512x512 /
REM 20 steps:  resident 1.34-1.44 s/image (peak 1.92 GB)
REM             cache    2.27-2.51 s/image (peak 1.34 GB)
REM -> resident is ~40% faster for only +0.58 GB.  Leave it on the default.
REM Only pass --te-mode cache if your card has under ~3 GB free.
"%PY%" app.py %*

echo.
echo UI exited. If this window flashed by, the reason is printed above.
pause
exit /b 0

:try
if "%PY%"=="" if exist %1 (
  %1 -c "import torch, transformers, diffusers, safetensors" >nul 2>&1
  if not errorlevel 1 set PY=%~1
)
exit /b 0
