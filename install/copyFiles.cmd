@echo off

echo xcopy /-Y /R /S starcraft C:\StarCraft
xcopy /-Y /R /S starcraft C:\StarCraft

if not "%ERRORLEVEL%" == "0" goto ERROR

goto QUIT

:ERROR
pause

:QUIT
