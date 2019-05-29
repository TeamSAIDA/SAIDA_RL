@echo off
setlocal enabledelayedexpansion
pushd "%~dp0"

set CURRENTDIR=%cd%

set RL_HOME_PATH=

for %%i in (1, 1, 10) do (
	cd ..
	set RL_HOME_PATH=!RL_HOME_PATH!..\
	IF EXIST .\protobuf\install\protoc.exe (
		popd

		for %%i in (*.proto) do (
			echo protoc.exe --cpp_out=. --python_out=. "%%i"
			!RL_HOME_PATH!\protobuf\install\protoc.exe --cpp_out=. --python_out=. "%%i"
			if not "!ERRORLEVEL!" == "0" goto ERROR2
		)
		for %%i in (*.pb.*) do (
			echo move /Y "%%i" !RL_HOME_PATH!\cpp\SAIDA\DeepLearning\message\
			move /Y "%%i" !RL_HOME_PATH!\cpp\SAIDA\DeepLearning\message\
			if not "!ERRORLEVEL!" == "0" goto ERROR2
		)
		for %%i in (*.py) do (
			echo move /Y "%%i" !RL_HOME_PATH!\python\saida_gym\envs\protobuf\
			move /Y "%%i" !RL_HOME_PATH!\python\saida_gym\envs\protobuf\
			if not "!ERRORLEVEL!" == "0" goto ERROR2
		)
		goto QUIT
	)
)

:ERROR
echo "protoc.exe 파일을 찾을 수 없습니다."

:ERROR2

pause
exit 1

:QUIT
echo 생성완료.
pause