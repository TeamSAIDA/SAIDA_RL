/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "SharedMemoryManager.h"

using namespace BWML;
using namespace MyBot;

SharedMemoryManager::~SharedMemoryManager() {
	for (auto s : shmList) {
		s->close();
	}

	shmList.clear();
}

SharedMemoryManager &SharedMemoryManager::Instance() {
	static SharedMemoryManager sharedMemoryManager;
	return sharedMemoryManager;
}

bool SharedMemoryManager::CreateMemoryMap(SharedMemory *shm)
{
	cout << "공유 메모리 생성 시작..(" << shm->getMapName() << ")" << endl;

	bool isCreateSAIDAIPC = false;

	// 기 오픈된 공유메모리가 있는지 확인.
	shm->hFileMap = OpenFileMapping(FILE_MAP_ALL_ACCESS, false, shm->getMapName());

	// 공유메모리 존재 안하는 경우 생성.
	if (!shm->hFileMap) {
		cout << "공유 메모리 새로 생성..." << endl;

		shm->hFileMap = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, shm->MAX_SHM_SIZE, shm->getMapName());

		// 생성 실패 시 종료 (메모리 부족, 보안 등)
		if (!shm->hFileMap) {
			Logger::error("공유메모리 생성 실패!");
			return false;
		}
		else {
			cout << "공유 메모리 새로 생성 성공" << endl;
			isCreateSAIDAIPC = true;
		}
	}
	else {
		cout << "공유 메모리 이미 존재" << endl;
	}

	if ((shm->pData = (char *)MapViewOfFile(shm->hFileMap, FILE_MAP_ALL_ACCESS, 0, 0, shm->MAX_SHM_SIZE)) == NULL) {
		Logger::error("공유메모리 닫기");
		CloseHandle(shm->hFileMap);
		return false;
	}
	else
	{
		// 내가 열었으면 초기화
		if (isCreateSAIDAIPC)
			memset(shm->pData, NULL, shm->MAX_SHM_SIZE);
	}

	shmList.push_back(shm);

	return true;
}

void SharedMemoryManager::FreeMemoryMap(SharedMemory *shm) {
	cout << "FreeMemoryMap" << endl;
	auto del = find_if(shmList.begin(), shmList.end(), [shm](SharedMemory * s) {
		return shm->getMapName() == s->getMapName();
	});

	if (del != shmList.end()) {
		if (shm && shm->hFileMap) {
			if (shm->pData)
				UnmapViewOfFile(shm->pData);

			if (shm->hFileMap)
				CloseHandle(shm->hFileMap);
		}

		shmList.erase(del);
	}
}
