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

	class RLAvoidReavers : public Gym
	{
	private:
		Unit agent = nullptr;

		int ACTION_TYPE;

		// 몇도씩 자를지를 결정
		int MOVE_ANGLE = 0;
		int DISTANCE = 0;
		int ACTION_SIZE = 0;

		Position GOAL_POS = Position(320, 320);
		vector<Position> MovePosition;
		Position commandPosition;
		int commandTime;

		bool startedWithOverlapped = false;
		bool isOverlapped = false;
		// step 당 프레임

		// Map Size
		const int map_size = 320;

		Position lastTargetPos;
		int lastAction = 0;
		bool invalidAction = false;
		int hitThreshHold = 0;

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
		RLAvoidReavers(string shmName, ConnMethod method = SHARED_MEMORY) : Gym("RLAvoidReavers") {
			connection = new RLSharedMemory(shmName, 2000000);
			connection->initialize();
		}
		~RLAvoidReavers() { };

		static RLAvoidReavers &Instance(string shmName = "");

		void step(::google::protobuf::Message *stepReqMsg) override;
		Position getValidPosition(Position from, Position direction);
		void render() override;

		void onUnitDestroy(Unit unit) override;
	};
}

