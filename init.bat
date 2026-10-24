for %%I in ("%~dp0..") do set "FOLDER_NAME=%%~nxI"
uv run tags.py
uv run assets.py --title "%FOLDER_NAME%"
pause