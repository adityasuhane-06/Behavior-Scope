@echo off
echo ============================================================
echo Behavior Scope - Quick Setup for Windows
echo ============================================================
echo.

echo [Step 1/3] Installing Python packages...
echo This may take several minutes...
echo.

pip install numpy>=1.24.0 scipy>=1.11.0 pandas>=2.0.0
pip install torch>=2.0.0 transformers>=4.35.0
pip install librosa>=0.10.0 pyannote.audio>=3.0.0
pip install opencv-python>=4.8.0 mediapipe>=0.10.0
pip install matplotlib>=3.7.0 seaborn>=0.12.0 plotly>=5.17.0
pip install scikit-learn>=1.3.0 pyyaml>=6.0

echo.
echo ============================================================
echo [Step 2/3] Checking installation...
echo ============================================================
python check_dependencies.py

echo.
echo ============================================================
echo [Step 3/3] Manual steps required:
echo ============================================================
echo.
echo 1. Install FFmpeg (for video processing):
echo    - Download from: https://ffmpeg.org/download.html
echo    - Or using winget: winget install ffmpeg
echo    - Or using chocolatey: choco install ffmpeg
echo.
echo 2. Get HuggingFace token (for speaker diarization):
echo    - Sign up at: https://huggingface.co/
echo    - Create token at: https://huggingface.co/settings/tokens
echo    - Accept pyannote terms: https://huggingface.co/pyannote/speaker-diarization-3.1
echo    - Set token: set HF_TOKEN=your_token_here
echo.
echo 3. Test the system:
echo    python main.py --video sample.mp4 --output results/
echo.
echo ============================================================
echo Setup complete!
echo ============================================================
pause
