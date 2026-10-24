for %%I in ("%~dp0..") do set "FOLDER_NAME=%%~nxI"
uv run photowall tags
uv run photowall assets --title "%FOLDER_NAME%"
pause
