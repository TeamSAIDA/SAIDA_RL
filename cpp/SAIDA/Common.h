/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include <cstdio>
#include <cstdlib>

#include <stdarg.h>
#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <deque>
#include <list>
#include <set>
#include <map>
#include <array>
#include <ctime>
#include <iomanip>

#include <winsock2.h>
#include <windows.h>

#include <BWAPI.h>

#include "Config.h"
#include "CommandUtil.h"
#include "BWEM/src/bwem.h"      // update the path if necessary

using namespace BWAPI;
using namespace BWAPI::UnitTypes;
using namespace std;
using namespace BWEM;
using namespace BWEM::BWAPI_ext;
using namespace BWEM::utils;

typedef unsigned int word;

namespace {
	auto &theMap = BWEM::BWEMMap::Instance();
	auto &bw = Broodwar;
}

#define S bw->self()
#define E bw->enemy()

#define TIME bw->getFrameCount()

#define GYM 1

namespace MyBot
{
	/// 로그 유틸
	namespace Logger
	{
		void appendTextToFile(const string &logFile, const string &msg);
		void appendTextToFile(const string &logFile, const char *fmt, ...);
		void overwriteToFile(const string &logFile, const string &msg);
		void debugFrameStr(const char *fmt, ...);
		void debug(const char *fmt, ...);
		void info(const string fileName, const bool printTime, const char *fmt, ...);
		void error(const char *fmt, ...);
	};

	class SAIDA_Exception
	{
	private:
		unsigned int nSE;
		PEXCEPTION_POINTERS     m_pException;
	public:
		SAIDA_Exception(unsigned int errCode, PEXCEPTION_POINTERS pException) : nSE(errCode), m_pException(pException) {}
		unsigned int getSeNumber() {
			return nSE;
		}
		PEXCEPTION_POINTERS getExceptionPointers() {
			return m_pException;
		}
	};

	/// 파일 유틸
	namespace FileUtil {
		/// 디렉토리 생성
		void MakeDirectory(const char *full_path);
		/// 파일 존재 체크 (createYn 가 true 이면 파일이 없는 경우 새로 생성)
		bool isFileExist(const char *filename, bool createYn = false);
		/// 파일 유틸 - 텍스트 파일을 읽어들인다
		string readFile(const string &filename);

		/// 파일 유틸 - 경기 결과를 텍스트 파일로부터 읽어들인다
		void readResults();

		/// 파일 유틸 - 경기 결과를 텍스트 파일에 저장한다
		void writeResults();

		/// fileName 파일의 startString 라인부터 endString 라인까지 삭제한다.
		void eraseHeader(char *fileName, char *startString, char *endString);
		void addHeader(char *fileName, vector<string> contents);
	}

	namespace CommonUtil {
		string getYYYYMMDDHHMMSSOfNow();
		void pause(int milli);
		void create_process(char *cmd, char *param = NULL, char *currentDirectory = NULL, bool isRelativePath = false);
		DWORD killProcessByName(char *processName);
		DWORD findProcessId(char *processName);
	}
}