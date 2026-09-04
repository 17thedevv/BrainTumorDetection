@echo off
echo ========================================================
echo DONG GOI UNG DUNG GUI VOI PYINSTALLER
echo ========================================================
echo.
echo Dang cai dat PyInstaller...
venv\Scripts\pip install pyinstaller

echo.
echo Dang dong goi ung dung (qua trinh nay co the mat vai phut do thu vien PyTorch rat nang)...
venv\Scripts\pyinstaller --noconfirm --onedir --windowed --name "BrainTumorDetection"  gui_app.py

echo.
echo ========================================================
echo HOAN TAT! 
echo Ung dung duoc luu trong thu muc: dist\BrainTumorDetection\
echo Ban co the chay file BrainTumorDetection.exe trong do.
echo ========================================================
pause
