@echo off
echo ================================================
echo Setting up Behavior Scope Environment
echo ================================================
echo.

echo Installing python-dotenv...
pip install python-dotenv

echo.
echo ✓ Setup complete!
echo.
echo The .env file has been created with your HuggingFace token.
echo You can now run the analysis:
echo.
echo   python main.py --video "data\raw\boystutter.mp4" --output "data\outputs"
echo.
pause
