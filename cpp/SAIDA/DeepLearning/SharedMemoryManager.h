/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once
#include "../Common.h"
#include "../InformationManager.h"
#include "../DeepLearning/Supervised.h"
#include "SharedMemory.h"

#define SHM	SharedMemoryManager::Instance()

namespace BWML
{
	class SharedMemoryManager
	{
	private:
		vector<SharedMemory *> shmList;

	public:
		~SharedMemoryManager();

		static SharedMemoryManager &Instance();

		bool CreateMemoryMap(SharedMemory *shm);
		void FreeMemoryMap(SharedMemory *shm);
	};
}
