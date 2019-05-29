/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once
#include "../Gym.h"
#include "../RLSharedMemory.h"

#define HUMAN_MODE 0

using namespace Message;

namespace BWML {
	class RLVultureVsZealot : public Gym
	{
	private:
		Unit agent = nullptr;
		vector<int> idOrderedList;
		vector<int> enemyIdOrderedList;

		int ACTION_TYPE;

		// 몇도씩 자를지를 결정
		int MOVE_ANGLE = 0;
		int DISTANCE = 0;
		int ACTION_SIZE = 0;

		int myKillCount = 0;
		int enemyKillCount = 0;

		int MY_UNIT_COUNT;
		int ENEMY_UNIT_COUNT;

		int lastAction = -1;
		Position lastTargetPos;

		vector<Position> MovePosition;
		Position commandPosition;
		int preCooldown;
		bool timeOver;

	protected:
		// Gym Override
		void init(::google::protobuf::Message *message) override;
		bool isDone() override;
		void reset(bool isFirstResetCall) override;
		bool isWin();
		bool isDefeat();
		float getReward() override;
		bool isResetFinished() override;
		bool isActionFinished() override;
		void makeInitMessage(::google::protobuf::Message *message) override;
		void getObservation(::google::protobuf::Message *stateMsg) override;

		bool initializeAndValidate() override;

	public:
		RLVultureVsZealot(string shmName, ConnMethod method = SHARED_MEMORY) : Gym("RLVultureVsZealot") {
			connection = new RLSharedMemory(shmName, 2000000);
			connection->initialize();
		}
		~RLVultureVsZealot() { };

		static RLVultureVsZealot &Instance(string shmName = "");

		void step(::google::protobuf::Message *stepReqMsg) override;
		void render() override;

		void onUnitDestroy(Unit unit) override;
	};
}

