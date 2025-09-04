@echo off
echo Starting Beyond Typing: Real-Time Conversational AI
echo.
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
echo.
echo All dependencies installed successfully!
echo.
echo Starting Streamlit app...
streamlit run app.py
pause
