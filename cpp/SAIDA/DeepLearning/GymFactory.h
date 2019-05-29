/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once
#include "Gym.h"
#include "SharedMemory.h"
#include "RLSharedMemory.h"

#include "message/common.pb.h"

#define MAX_GYM_NAME_LENGTH 30 // Gym 이름의 최대 자릿수

namespace BWML {
	enum class AIType {
		EMBEDED, DLL, EXE, HUMAN
	};

	class GymFactory
	{
	private:
		Gym *gym;
		// TODO 추후 namespace 는 conn 등으로 수정. sharedmemory 와 zmq 등을 자식으로 하는 부모 클래스로 변경 필요.
		SharedMemory *connection;

		// Gym 이 추가되면 변경되어야 할 메소드
		void String2Gym(string gymName, string shmName, int version);
		string mapName;
		string autoMenuMode = "SINGLE_PLAYER";
		string enemyBot = "";
		AIType enemyType = AIType::EMBEDED;

		bool autoKillStarcraft = true;

	public:
		GymFactory() {
			gym = nullptr;
		}
		~GymFactory() {}

		static GymFactory &Instance();

		void Initialize(ConnMethod method = SHARED_MEMORY);

		Gym *GetGym() {
			return gym;
		}

		void InitializeGym() {
			gym->initialize();
		}

		void Destroy();
	};
}

