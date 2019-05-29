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

#define PTEST 0
#define INVALID_REWARD 0

namespace BWML {
	class RLZealotVsZealot : public Gym
	{
	private:
		vector<Unit> myUnitSet;
		vector<Unit> enUnitSet;

		int ACTION_TYPE;

		// 몇도씩 자를지를 결정
		int MOVE_ANGLE = 0;
		int DISTANCE = 0;
		int ACTION_SIZE = 0;
		int ACTION_MOVE = 0;
		int ACTION_ATTACK = 0;
		int ACTION_HOLD = 0;
		int ACTION_NOTHING = 0;

		const int MY_UNIT_COUNT = 3;
		const int ENEMY_UNIT_COUNT = 3;

		Position lastTargetPos;

		vector<Position> MovePosition;
		Position commandPosition;

		bool invalidAction;

		UnitType myType;
		UnitType enType;

		int lastAction;

		vector<vector<vector<int>>> renderInfo;

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
		RLZealotVsZealot(string shmName, ConnMethod method = SHARED_MEMORY) : Gym("RLZealotVsZealot") {
			connection = new RLSharedMemory(shmName, 2000000);
			connection->initialize();
		}
		~RLZealotVsZealot() { };

		static RLZealotVsZealot &Instance(string shmName = "");

		void step(::google::protobuf::Message *stepReqMsg) override;
		void render() override;

		void onUnitDestroy(Unit unit) override;
	};
}
