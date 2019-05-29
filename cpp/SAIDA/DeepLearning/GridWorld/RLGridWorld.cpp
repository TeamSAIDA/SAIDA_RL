/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "RLGridWorld.h"
#include "../../InformationManager.h"
#include <cstdlib>

using namespace MyBot;

RLGridWorld::RLGridWorld(string shmName, ConnMethod method) : Gym("RLGridWorld")
{
	// 1. connect
	connection = new RLSharedMemory(shmName, 2000000);
	connection->initialize();

	Message::GridWorld::InitRes initRes;
	initResMsg = initRes.New();
	Message::GridWorld::ResetRes resetRes;
	resetResMsg = resetRes.New();
	Message::GridWorld::StepRes stepRes;
	stepResMsg = stepRes.New();
	Message::GridWorld::RenderReq renderReq;
	renderReqMsg = renderReq.New();

	q_table = vector<vector<float>>(25, vector<float>(4, 0.0));
}

RLGridWorld &RLGridWorld::Instance(string shmName) {
	static RLGridWorld rlGridWorld(shmName);
	return rlGridWorld;
}

void RLGridWorld::init(::google::protobuf::Message *message) {
	Message::InitReq *initReq = (Message::InitReq *)message;
	STEP_FRAME = initReq->frames_per_step();

	startPos = theMap.Center() / BIG_TILEPOSITION_SCALE * BIG_TILEPOSITION_SCALE + BIG_TILEPOSITION_SCALE / 2;
	leftTop = (BigTilePosition)startPos;
}

void RLGridWorld::reset(bool isFirstResetCall)
{
	agentPos = Positions::Origin;
	action = -1;
	agent = nullptr;
}

bool RLGridWorld::initializeAndValidate()
{
	// 유닛이 존재하는 경우에만 동작
	if (!agent || !agent->exists()) {
		uList units = INFO.getUnits(Terran_Vulture, S);

		if (units.empty())
			return false;

		agent = units.at(0)->unit();
	}

	return true;
}

void RLGridWorld::step(::google::protobuf::Message *stepReqMsg)
{
	Message::Action act = ((Message::StepReq *)stepReqMsg)->action(0);
	int action = act.action_num();

	if (action == 0)  //상
		agentPos -= Position(0, 1);
	else if (action == 1) // 하
		agentPos += Position(0, 1);
	else if (action == 2)  // 좌
		agentPos -= Position(1, 0);
	else if (action == 3)  // 우
		agentPos += Position(1, 0);

	makeValid(agentPos);

	CommandUtil::move(agent, index2Position(agentPos));

	return;
}

bool RLGridWorld::isDone()
{
	return agentPos == GOAL || agentPos == TRAP[0] || agentPos == TRAP[1];
}

float RLGridWorld::getReward() {
	if (agentPos == GOAL)
		return 1;
	else if (agentPos == TRAP[0] || agentPos == TRAP[1])
		return -1;

	return 0;
}

void RLGridWorld::getObservation(::google::protobuf::Message *stateMsg)
{
	Message::GridWorld::State *stateMessage = (Message::GridWorld::State *)stateMsg;

	stateMessage->set_index(to_string(index2order(agentPos)));
}

bool RLGridWorld::isResetFinished()
{
	return (BigTilePosition)agent->getPosition() == (BigTilePosition)startPos;
}

bool RLGridWorld::isActionFinished()
{
	if (agent->getPosition() == index2Position(agentPos))
		return true;
	else {
		CommandUtil::move(agent, index2Position(agentPos));
		return false;
	}
}

void RLGridWorld::makeInitMessage(::google::protobuf::Message *initMessage)
{
	Message::GridWorld::InitRes *message = (Message::GridWorld::InitRes *)initMessage;
	message->set_max_row(5);
	message->set_max_col(5);
	message->set_num_action_space(4);
	message->set_map("SFFFFFFHFFFHGFFFFFFFFFFFF");
}

void RLGridWorld::makeResetResultMessage(::google::protobuf::Message *message)
{
	Message::GridWorld::ResetRes *stepMessage = (Message::GridWorld::ResetRes *)message;
	getObservation(stepMessage->mutable_next_state());
}

void RLGridWorld::makeStepResultMessage(::google::protobuf::Message *message)
{
	Message::GridWorld::StepRes *stepMessage = (Message::GridWorld::StepRes *)message;
	stepMessage->set_done(isDone());
	stepMessage->set_reward(getReward());
	getObservation(stepMessage->mutable_next_state());
}

void RLGridWorld::makeValid(Position &pos)
{
	if (pos.x < 0) pos.x = 0;
	else if (pos.x >= MAX_X) pos.x = MAX_X - 1;

	if (pos.y < 0) pos.y = 0;
	else if (pos.y >= MAX_Y) pos.y = MAX_Y - 1;
}

Position RLGridWorld::index2Position(Position pos)
{
	// 시작위치 + index * BIG_TILEPOSITION_SCALE
	return startPos + pos * BIG_TILEPOSITION_SCALE;
}

void RLGridWorld::setRenderData(::google::protobuf::Message *message) {
	Message::GridWorld::RenderReq *renderMessage = (Message::GridWorld::RenderReq *)message;

	for (unsigned int i = 0; i < q_table.size(); i++) {
		for (unsigned int j = 0; j < q_table[i].size(); j++) {
			if (renderMessage->q_table_size() <= int(i * q_table[i].size() + j))
				q_table[i][j] = 0;
			else
				q_table[i][j] = renderMessage->q_table(i * q_table[i].size() + j);
		}
	}
}

void RLGridWorld::render()
{
	UnitInfo *self = INFO.getUnitInfo(agent, S);

	if (self)
		focus(self->pos());
	else
		focus(startPos);

	// 행
	for (int i = 0; i <= MAX_Y; i++)
		bw->drawLineMap((Position)(leftTop + BigTilePosition(0, i)),
						(Position)(leftTop + BigTilePosition(MAX_X, i)), Colors::White);

	// 열
	for (int i = 0; i <= MAX_X; i++)
		bw->drawLineMap((Position)(leftTop + BigTilePosition(i, 0)),
						(Position)(leftTop + BigTilePosition(i, MAX_Y)), Colors::White);

	// 칸별 웨이트 값 출력
	for (int i = 0; i < MAX_X * MAX_Y; i++) {
		int r = i / MAX_Y;
		int c = i % MAX_X;

		Position center = (Position)BigTilePosition(leftTop.x + c, leftTop.y + r) + Position(48, 48);
		bw->drawTextMap(center - Position(0, 48), "%.2f", q_table[i][0]);
		bw->drawTextMap(center + Position(0, 35), "%.2f", q_table[i][1]);
		bw->drawTextMap(center - Position(48, 0), "%.2f", q_table[i][2]);
		bw->drawTextMap(center + Position(28, 0), "%.2f", q_table[i][3]);
	}

	// 세모
	Position trap1 = (Position)BigTilePosition(leftTop.x + 2, leftTop.y + 1) + 48;
	Position trap2 = (Position)BigTilePosition(leftTop.x + 1, leftTop.y + 2) + 48;

	bw->drawCircleMap((Position)trap1, 16, Colors::Red, true);
	bw->drawCircleMap((Position)trap2, 16, Colors::Red, true);

	// 동그라미
	Position goal = (Position)BigTilePosition(leftTop.x + 2, leftTop.y + 2) + 48;
	bw->drawCircleMap((Position)goal, 16, Colors::Green, true);
}