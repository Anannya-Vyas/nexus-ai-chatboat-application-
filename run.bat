@echo off
echo Installing required packages...
pip install streamlit groq requests pillow PyMuPDF python-docx duckduckgo-search
echo.
echo Starting NexusAI...
streamlit run app.py
pause
