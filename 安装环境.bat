@echo off
echo Setting up environment...

:: Install required Python packages
pip install opencv-python-headless
pip install transformers
pip install pillow
pip install torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: Verify installation
if %ERRORLEVEL% NEQ 0 (
    echo Error occurred during package installation. Exiting...
    exit /b 1
)

echo Environment setup complete.
pause
