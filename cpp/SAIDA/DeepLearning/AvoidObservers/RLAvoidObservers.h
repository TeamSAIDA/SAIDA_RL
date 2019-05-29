/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "../Gym.h"
#include "../RLSharedMemory.h"

using namespace Message;

#pragma once
namespace BWML {
	class RLAvoidObservers : public Gym
	{
	private:
		Unit agent = nullptr;

		int ACTION_TYPE;

		// 몇도씩 자를지를 결정
		int MOVE_ANGLE = 0;
		int DISTANCE = 0;
		int ACTION_SIZE = 0;

		int OBSERVABLE_TILE_SPACE_RADIUS = 10;

		vector<Position> MovePosition;
		Position commandPosition;
		int commandTime;

		// step 당 프레임
		Position lastTargetPos;
		int lastAction;
		Position destroyPos = Positions::None; // 마지막으로 죽은 Position
	protected:
		// Gym Override
		void init(::google::protobuf::Message *message) override;
		bool isDone() override;
		bool isLogicalDone();
		void reset(bool isFirstResetCall) override;
		float getReward() override;
		bool isResetFinished() override;
		bool isActionFinished() override;
		void makeInitMessage(::google::protobuf::Message *message) override;
		void getObservation(::google::protobuf::Message *stateMsg) override;

		bool initializeAndValidate() override;

	public:
		RLAvoidObservers(string shmName, ConnMethod method = SHARED_MEMORY) : Gym("RLAvoidObservers") {
			connection = new RLSharedMemory(shmName, 2000000);
			connection->initialize();
		}
		~RLAvoidObservers() { };

		static RLAvoidObservers &Instance(string shmName = "");

		void step(::google::protobuf::Message *stepReqMsg) override;
		void render() override;

		void onUnitDestroy(Unit unit) override;
	};
}

