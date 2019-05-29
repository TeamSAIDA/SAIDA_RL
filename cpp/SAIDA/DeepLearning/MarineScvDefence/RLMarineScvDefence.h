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
#include "../message/marineScv.pb.h"

#define HUMAN_MODE 0

using namespace Message::MarineScv;

namespace BWML {
	class RLMarineScvDefence : public Gym
	{
	private:
		vector<int> idOrderedMarineList;
		vector<int> idOrderedScvList;
		vector<int> enemyIdOrderedList;

		int ACTION_TYPE;

		// 몇도씩 자를지를 결정
		int MOVE_ANGLE = 0;
		int DISTANCE = 0;
		int ACTION_SIZE = 0;

		int myKillCount = 0;
		int enemyKillCount = 0;

		int MY_UNIT_COUNT[2];
		int ENEMY_UNIT_COUNT;

		UnitType myUnitType[2];
		UnitType enUnitType;

		int lastAction = -1;
		Position lastTargetPos;

		vector<Position> MovePosition;
		Position commandPosition;
		int commandTime;
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
		RLMarineScvDefence(string shmName, ConnMethod method = SHARED_MEMORY) : Gym("RLMarineScvDefence") {
			connection = new RLSharedMemory(shmName, 2000000);
			connection->initialize();

			Message::MarineScv::InitReq initReq;
			initReqMsg = initReq.New();
			Message::MarineScv::InitRes initRes;
			initResMsg = initRes.New();
			Message::MarineScv::ResetRes resetRes;
			resetResMsg = resetRes.New();
			Message::MarineScv::StepReq stepReq;
			stepReqMsg = stepReq.New();
			Message::MarineScv::StepRes stepRes;
			stepResMsg = stepRes.New();
		}
		~RLMarineScvDefence() { };

		static RLMarineScvDefence &Instance(string shmName);




		void step(::google::protobuf::Message *stepReqMsg) override;
		void render() override;

		void onUnitDestroy(Unit unit) override;
	};
}

