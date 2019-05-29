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

namespace BWML {

	class RLAvoidZerglings : public Gym
	{
	private:
		Unit agent = nullptr;
		UnitType agent_type = Terran_SCV;
		int ACTION_TYPE;

		// 몇도씩 자를지를 결정
		int MOVE_ANGLE = 0;
		int DISTANCE = 0;
		int ACTION_SIZE = 0;

		vector<Position> MovePosition;
		Position commandPosition;
		int commandTime;

		Position lastTargetPos;
		int lastAction = 0;
		Position destroyPos = Positions::None; // 마지막으로 죽은 Position
		float lastReward = 0;
		int lastGas = 0;

		double EPSILON = 0.00000001;
		double RADIAN_UNIT = 0;

		int direction = 1;

	protected:
		// Gym Override
		void init(::google::protobuf::Message *message) override;
		bool isDone() override;
		void reset(bool isFirstResetCall) override;
		float getReward() override;
		bool isResetFinished() override;
		bool isActionFinished() override;
		void makeInitMessage(::google::protobuf::Message *message) override;
		void getObservation(::google::protobuf::Message *stateMsg) override;
		bool initializeAndValidate() override;
	public:
		RLAvoidZerglings(string shmName, ConnMethod method = SHARED_MEMORY) : Gym("RLAvoidZerglings") {
			connection = new RLSharedMemory(shmName, 2000000);
			connection->initialize();
		}
		~RLAvoidZerglings() { };

		static RLAvoidZerglings &Instance(string shmName = "");

		void step(::google::protobuf::Message *stepReqMsg) override;
		void render() override;

		void onUnitDestroy(Unit unit) override;
	};
}

