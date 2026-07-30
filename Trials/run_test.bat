@echo off
cd /d "C:\Users\ericb\Documents\GitHub\PSD_opt"
echo ============================================
echo NUKLEATION TEST WIRD AUSGEFUEHRT...
echo ============================================
echo.

REM Versuche verschiedene Python-Installationen
if exist "C:\Program Files\Python311\python.exe" (
    "C:\Program Files\Python311\python.exe" Trials\testcase_nucleation_all_methods.py
    goto :end
)

if exist "C:\Program Files\Python310\python.exe" (
    "C:\Program Files\Python310\python.exe" Trials\testcase_nucleation_all_methods.py
    goto :end
)

if exist "C:\Program Files\Python39\python.exe" (
    "C:\Program Files\Python39\python.exe" Trials\testcase_nucleation_all_methods.py
    goto :end
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" Trials\testcase_nucleation_all_methods.py
    goto :end
)

if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" Trials\testcase_nucleation_all_methods.py
    goto :end
)

REM Versuche py Launcher (Windows Standard)
py --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py Trials\testcase_nucleation_all_methods.py
    goto :end
)

REM Fallback: python im PATH
python Trials\testcase_nucleation_all_methods.py
goto :end

:end
echo.
echo ============================================
echo TEST ABGESCHLOSSEN
echo ============================================
echo.
pause
