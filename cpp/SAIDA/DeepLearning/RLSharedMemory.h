/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once

#include "../InformationManager.h"
#include "SharedMemory.h"
#include "GymFactory.h"

namespace BWML {
	class RLSharedMemory : public SharedMemory
	{
	protected:
		char *receiveHandler(char *message) override;

	public:
		RLSharedMemory(string name, size_t size = 2000000) : SharedMemory(name, size) { }
	};
}
