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

using namespace Message;

#define HUMAN_MODE 0

namespace BWML {
	class RLMarineVsZergling : public Gym
	{
	private:
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

		vector<Position> MovePosition;
		Position commandPosition;
		int commandTime;
		bool timeOver;

	protected:
		// Gym Override
		void init(::google::protobuf::Message *message) override;
		bool isDone() override;
		void reset(bool isFirstResetCall) override;
		bool isWin();
		bool isDefeat();
		bool isTimeout();
		float getReward() override;
		bool isResetFinished() override;
		bool isActionFinished() override;
		void makeInitMessage(::google::protobuf::Message *message) override;
		void getObservation(::google::protobuf::Message *stateMsg) override;

		bool initializeAndValidate() override;

	public:
		RLMarineVsZergling(string shmName, ConnMethod method = SHARED_MEMORY) : Gym("MarineVsZergling") {
			connection = new RLSharedMemory(shmName, 2000000);
			connection->initialize();
		}
		~RLMarineVsZergling() { };

		static RLMarineVsZergling &Instance(string shmName = "");

		void step(::google::protobuf::Message *stepReqMsg) override;
		void render() override;

		void onUnitDestroy(Unit unit) override;
	};
}

