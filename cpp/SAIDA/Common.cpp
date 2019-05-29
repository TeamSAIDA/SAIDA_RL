/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "Common.h"
#include "windows.h"
#include "direct.h"
#include <tlhelp32.h>

using namespace MyBot;

void Logger::appendTextToFile(const string &logFile, const string &msg)
{
	ofstream logStream;
	logStream.open(logFile.c_str(), ofstream::app);
	logStream << msg;
	logStream.flush();
	logStream.close();
}

void Logger::appendTextToFile(const string &logFile, const char *fmt, ...)
{
	va_list arg;

	va_start(arg, fmt);
	//vfprintf(log_file, fmt, arg);
	char buff[256];
	vsnprintf_s(buff, 256, fmt, arg);
	va_end(arg);

	ofstream logStream;
	logStream.open(logFile.c_str(), ofstream::app);
	logStream << buff;
	logStream.flush();
	logStream.close();
}

void Logger::overwriteToFile(const string &logFile, const string &msg)
{
	ofstream logStream(logFile.c_str());
	logStream << msg;
	logStream.flush();
	logStream.close();
}

void Logger::debugFrameStr(const char *fmt, ...) {
#ifdef _LOGGING

	//if (TIME < 12000 || TIME > 14000)
	//	return;

	va_list arg;

	va_start(arg, fmt);
	//vfprintf(log_file, fmt, arg);
	char buff[256];
	vsnprintf_s(buff, 256, fmt, arg);
	va_end(arg);

	ofstream logStream;
	logStream.open(Config::Files::WriteDirectory + Config::Files::LogFilename.c_str(), ofstream::app);
	logStream << "[" << TIME << "] ";
	logStream << buff;
	logStream.flush();
	logStream.close();
#endif
}

void Logger::debug(const char *fmt, ...) {
#ifdef _LOGGING

	//if (TIME < 12000 || TIME > 14000)
	//	return;

	va_list arg;

	va_start(arg, fmt);
	//vfprintf(log_file, fmt, arg);
	char buff[256];
	vsnprintf_s(buff, 256, fmt, arg);
	va_end(arg);

	ofstream logStream;
	logStream.open(Config::Files::WriteDirectory + Config::Files::LogFilename.c_str(), ofstream::app);
	logStream << buff;
	logStream.flush();
	logStream.close();
#endif
}

void Logger::info(const string fileName, const bool printTime, const char *fmt, ...) {
	va_list arg;

	va_start(arg, fmt);
	//vfprintf(log_file, fmt, arg);
	char buff[512];
	vsnprintf_s(buff, 512, fmt, arg);
	va_end(arg);

	ofstream logStream;
	logStream.open(Config::Files::WriteDirectory + fileName, ofstream::app);

	if (printTime)
		logStream << "[" << TIME << "] ";

	logStream << buff;
	logStream.flush();
	logStream.close();
}

void Logger::error(const char *fmt, ...) {
	va_list arg;

	va_start(arg, fmt);
	//vfprintf(log_file, fmt, arg);
	char buff[256];
	vsnprintf_s(buff, 256, fmt, arg);
	va_end(arg);

	ofstream logStream;
	logStream.open(Config::Files::WriteDirectory + Config::Files::ErrorLogFilename.c_str(), ofstream::app);
	logStream << "[" << TIME << "] ";
	logStream << buff;
	logStream.flush();
	logStream.close();
}

void FileUtil::MakeDirectory(const char *full_path)
{
	char temp[256], *sp;
	strcpy_s(temp, sizeof(temp), full_path);
	sp = temp; // 포인터를 문자열 처음으로

	while ((sp = strchr(sp, '\\'))) { // 디렉토리 구분자를 찾았으면
		if (sp > temp && *(sp - 1) != ':') { // 루트디렉토리가 아니면
			*sp = '\0'; // 잠시 문자열 끝으로 설정
			//mkdir(temp, S_IFDIR);
			CreateDirectory(temp, NULL);
			// 디렉토리를 만들고 (존재하지 않을 때)
			*sp = '\\'; // 문자열을 원래대로 복귀
		}

		sp++; // 포인터를 다음 문자로 이동
	}
}

bool MyBot::FileUtil::isFileExist(const char *filename, bool createYn)
{
	ifstream ifileStream;
	ofstream ofileStream;

	ifileStream.open(filename, ios::in);

	if (!ifileStream.is_open()) {
		if (createYn) {
			ofileStream.open(filename, ios::out);
			ofileStream.close();
		}

		return false;
	}

	ifileStream.close();

	return true;
}

string FileUtil::readFile(const string &filename)
{
	stringstream ss;

	FILE *file;
	errno_t err;

	if ((err = fopen_s(&file, filename.c_str(), "r")) != 0)
	{
		cout << "Could not open file: " << filename.c_str();
	}
	else
	{
		char line[4096]; /* or other suitable maximum line size */

		while (fgets(line, sizeof line, file) != nullptr) /* read a line */
		{
			ss << line;
		}

		fclose(file);
	}

	return ss.str();
}

void FileUtil::readResults()
{
	string enemyName = BWAPI::Broodwar->enemy()->getName();
	replace(enemyName.begin(), enemyName.end(), ' ', '_');

	string enemyResultsFile = Config::Files::ReadDirectory + enemyName + ".txt";

	//int wins = 0;
	//int losses = 0;

	FILE *file;
	errno_t err;

	if ((err = fopen_s(&file, enemyResultsFile.c_str(), "r")) != 0)
	{
		cout << "Could not open file: " << enemyResultsFile.c_str();
	}
	else
	{
		char line[4096]; /* or other suitable maximum line size */

		while (fgets(line, sizeof line, file) != nullptr) /* read a line */
		{
			//stringstream ss(line);
			//ss >> wins;
			//ss >> losses;
		}

		fclose(file);
	}
}

void FileUtil::writeResults()
{
	string enemyName = BWAPI::Broodwar->enemy()->getName();
	replace(enemyName.begin(), enemyName.end(), ' ', '_');

	string enemyResultsFile = Config::Files::WriteDirectory + enemyName + ".txt";

	stringstream ss;

	//int wins = 1;
	//int losses = 0;

	//ss << wins << " " << losses << "\n";

	Logger::overwriteToFile(enemyResultsFile, ss.str());
}

void MyBot::FileUtil::eraseHeader(char *fileName, char *startString, char *endString)
{
	ifstream ifileStream;
	ofstream ofileStream;

	ifileStream.open(fileName, ios::in);

	if (!ifileStream.is_open()) {
		ofileStream.open(fileName, ios::out);
		ofileStream.close();
		return;
	}

	char backup_filename[100];

	strcpy_s(backup_filename, sizeof(backup_filename), fileName);
	strcat_s(backup_filename, sizeof(backup_filename), ".bak");

	ofileStream.open(backup_filename, ios::out);

	char buffer[512];

	ifileStream.getline(buffer, sizeof(buffer));

	const char *HEADER = startString;

	while (!ifileStream.eof() && strstr(buffer, HEADER) == nullptr) {
		ofileStream << buffer << endl;
		ifileStream.getline(buffer, sizeof(buffer));
	}

	const char *FOOTER = endString;

	while (!ifileStream.eof()) {
		if (strstr(buffer, FOOTER) != nullptr) {
			ifileStream.getline(buffer, sizeof(buffer));
			break;
		}

		ifileStream.getline(buffer, sizeof(buffer));
	}

	while (!ifileStream.eof()) {
		ofileStream << buffer << endl;
		ifileStream.getline(buffer, sizeof(buffer));
	}

	ofileStream.flush();
	ifileStream.close();
	ofileStream.close();

	remove(fileName);
	rename(backup_filename, fileName);
}

void MyBot::FileUtil::addHeader(char *fileName, vector<string> contents) {
	ifstream ifileStream;
	ofstream ofileStream;

	ifileStream.open(fileName, ios::in);

	if (!ifileStream.is_open()) {
		ofileStream.open(fileName, ios::out);
		ofileStream.close();
		return;
	}

	char backup_filename[100];

	strcpy_s(backup_filename, sizeof(backup_filename), fileName);
	strcat_s(backup_filename, sizeof(backup_filename), ".bak");

	ofileStream.open(backup_filename, ios::out);

	for (auto content : contents)
		ofileStream << content << endl;

	char buffer[512];

	ifileStream.getline(buffer, sizeof(buffer));

	while (!ifileStream.eof()) {
		ofileStream << buffer << endl;
		ifileStream.getline(buffer, sizeof(buffer));
	}

	ofileStream.flush();
	ifileStream.close();
	ofileStream.close();

	remove(fileName);
	rename(backup_filename, fileName);
}

string CommonUtil::getYYYYMMDDHHMMSSOfNow()
{
	time_t timer;
	tm timeinfo;
	char buffer[80];
	timer = time(NULL);
	localtime_s(&timeinfo, &timer);
	strftime(buffer, 80, "%Y-%m-%d-%H-%M-%S", &timeinfo);
	return buffer;
}

void CommonUtil::pause(int milli) {
	Broodwar->pauseGame();
	Sleep(milli);
	Broodwar->resumeGame();
}

void CommonUtil::create_process(char *cmd, char *param, char *currentDirectory, bool isRelativePath) {
	STARTUPINFO StartupInfo = STARTUPINFO();
	PROCESS_INFORMATION ProcessInfo;
	StartupInfo.cb = sizeof(StartupInfo);

	BOOL ret;

	if (isRelativePath)
		ret = CreateProcess(NULL, (param == NULL ? cmd : (char *)((string)cmd + " " + param).c_str()), NULL, NULL, FALSE, CREATE_NEW_CONSOLE | NORMAL_PRIORITY_CLASS, NULL, currentDirectory, &StartupInfo, &ProcessInfo);
	else {
		// (char *)(path + run).c_str()
		ret = CreateProcess((char *)((string)currentDirectory + cmd).c_str(), (param == NULL ? cmd : (char *)((string)cmd + " " + param).c_str()), NULL, NULL, FALSE, CREATE_NEW_CONSOLE | NORMAL_PRIORITY_CLASS, NULL, currentDirectory, &StartupInfo, &ProcessInfo);
	}

	if (!ret) {
		if (isRelativePath) {
			char strBuffer[_MAX_PATH] = { 0, };
			char *pstrBuffer = NULL;

			pstrBuffer = _getcwd(strBuffer, _MAX_PATH);

			cout << "ErrorCode : " << GetLastError() << " (" << pstrBuffer << " " << cmd << ")" << endl;
			Logger::error("Create Process Error ErrorCode : %d (%s %s)", GetLastError(), pstrBuffer, cmd);
		}
		else {
			cout << "ErrorCode : " << GetLastError() << " (" << cmd << ")" << endl;
		}
	}
	else {
		cout << "Process ID = " << ProcessInfo.dwProcessId << endl;
		cout << "success." << endl;
	}
}

DWORD CommonUtil::killProcessByName(char *processName) {
	// 모든 프로세스의 정보 스냅샷.
	HANDLE h_processSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPALL, NULL);

	if (h_processSnapshot == INVALID_HANDLE_VALUE)
		return 0;

	PROCESSENTRY32 pEntry;
	pEntry.dwSize = sizeof(pEntry);

	Process32First(h_processSnapshot, &pEntry);

	bool killYn = false;

	do {
		if (!strcmp(processName, pEntry.szExeFile)) {
			// 종료를 위해 모든 액세스 권한을 받아온다.
			HANDLE h_KillProcess = OpenProcess(PROCESS_TERMINATE, FALSE, pEntry.th32ProcessID);

			if (h_KillProcess) {
				if (TerminateProcess(h_KillProcess, 0)) {
					unsigned long nCode;
					GetExitCodeProcess(h_KillProcess, &nCode);

					cout << "process (id : " << pEntry.th32ProcessID << ") is killed!" << endl;

					killYn = true;
				}
				else {
					cout << "process kill failed.." << GetLastError() << endl;
				}

				CloseHandle(h_KillProcess);
			}
			else {
				cout << "permission fail. " << GetLastError() << endl;
			}
		}
	} while (Process32Next(h_processSnapshot, &pEntry));

	CloseHandle(h_processSnapshot);

	if (killYn)
		cout << "killed all " << processName << " processes." << endl;

	return 0;
}


DWORD CommonUtil::findProcessId(char *processName) {
	// 모든 프로세스의 정보 스냅샷.
	HANDLE h_processSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPALL, NULL);

	if (h_processSnapshot == INVALID_HANDLE_VALUE)
		return 0;

	PROCESSENTRY32 pEntry;
	pEntry.dwSize = sizeof(pEntry);

	Process32First(h_processSnapshot, &pEntry);

	do {
		if (!strcmp(processName, pEntry.szExeFile)) {
			CloseHandle(h_processSnapshot);
			return pEntry.th32ProcessID;
		}
	} while (Process32Next(h_processSnapshot, &pEntry));

	CloseHandle(h_processSnapshot);

	return 0;
}
