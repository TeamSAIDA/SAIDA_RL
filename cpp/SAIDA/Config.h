/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once

#include "BWAPI.h"
#include <cassert>

// minwindef 정의
#define MAX_PATH 260

/// 봇 프로그램 설정
namespace Config
{
	/// 파일 관련 설정
	namespace Files
	{
		/// 로그 파일 이름
		extern std::string LogFilename;
		/// 타임아웃 파일 이름
		extern std::string TimeoutFilename;
		/// 에러로그 파일 이름
		extern std::string ErrorLogFilename;
		/// 읽기 파일 경로
		extern std::string ReadDirectory;
		/// 쓰기 파일 경로
		extern std::string WriteDirectory;
		/// SAIDA.exe 파일이 있는 폴더
		extern char saidaDirectory[MAX_PATH];
		/// saida.ini path
		extern char saida_ini_filename[MAX_PATH];
		/// Starcraft HOME 폴더
		extern char StarcraftDirectory[MAX_PATH];
		/// bwapi.ini path
		extern char bwapi_ini_filename[MAX_PATH];

		void initialize();
	}

	/// CommonUtil 관련 설정
	namespace Tools
	{
		/// MapGrid 에서 한 개 GridCell 의 size
		extern int MAP_GRID_SIZE;
	}

	/// BWAPI 옵션 관련 설정
	namespace BWAPIOptions
	{
		/// 로컬에서 게임을 실행할 때 게임스피드 (코드 제출 후 서버에서 게임을 실행할 때는 서버 설정을 사용함)<br>
		/// Speedups for automated play, sets the number of milliseconds bwapi spends in each frame.<br>
		/// Fastest: 42 ms/frame.  1초에 24 frame. 일반적으로 1초에 24frame을 기준 게임속도로 합니다.<br>
		/// Normal: 67 ms/frame. 1초에 15 frame.<br>
		/// As fast as possible : 0 ms/frame. CPU가 할수있는 가장 빠른 속도.
		extern int SetLocalSpeed;
		/// 로컬에서 게임을 실행할 때 FrameSkip (코드 제출 후 서버에서 게임을 실행할 때는 서버 설정을 사용함)<br>
		/// frameskip을 늘리면 화면 표시도 업데이트 안하므로 훨씬 빠릅니다
		extern int SetFrameSkip;
		/// rendering on/off
		extern bool EnableGui;
		/// 로컬에서 게임을 실행할 때 사용자 키보드/마우스 입력 허용 여부 (코드 제출 후 서버에서 게임을 실행할 때는 서버 설정을 사용함)
		extern bool EnableUserInput;
		/// 로컬에서 게임을 실행할 때 전체 지도를 다 보이게 할 것인지 여부 (코드 제출 후 서버에서 게임을 실행할 때는 서버 설정을 사용함)
		extern bool EnableCompleteMapInformation;
		/// 게임을 재시작 가능하게 해줌.
		extern bool RestartGame;
		/// 게임 종료(leaveGame) 후에 추가로 수행되는 on frame 안돌도록 함.
		extern bool EndGame;
	}

	/// 디버그 관련 설정
	namespace Debug
	{
		/// 화면 표시 여부 - 게임 정보
		extern bool DrawGameInfo;

		/// 화면 표시 여부 - 지도
		extern bool DrawBWEMInfo;

		/// 화면 표시 여부 - 유닛 ~ Target 간 직선
		extern bool DrawUnitTargetInfo;



		/// 화면 표시 여부 - 정찰 상태
		extern bool DrawScoutInfo;

		/// 화면 표시 여부 - 마우스 커서
		extern bool DrawMouseCursorInfo;

		/// 화면 표시 여부 - AllUnitVector, Map Information
		extern bool DrawMyUnit;
		extern bool DrawEnemyUnit;

		extern bool DrawLastCommandInfo;
		extern bool DrawUnitStatus;

		extern bool Focus;
		extern bool Console_Log;
	}

	// 기본 설정 정보
	namespace Propeties
	{
		/// 자원 측정시에 사용되는 measure duration 정보이며 단위는 seconds
		extern int duration;
		extern bool recoring;

	}

}