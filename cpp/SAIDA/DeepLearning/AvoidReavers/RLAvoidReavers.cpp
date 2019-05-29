/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "RLAvoidReavers.h"
#include "../../UXManager.h"

using namespace BWML;
using namespace MyBot;

RLAvoidReavers &RLAvoidReavers::Instance(string shmName) {
	static RLAvoidReavers instance(shmName);
	return instance;
}

//// INIT 관련
void RLAvoidReavers::init(::google::protobuf::Message *message)
{
	InitReq *initReq = (InitReq *)message;

	int map_version = initReq->version();

	// ACTION TYPE : 0 -> Action Number
	// ACTION TYPE : 1 -> Position X, Y Action Number
	// ACTION TYPE : 2 -> Angle, Radius, Action Number
	// ACTION NUMBER : 0 - Move, 1 - Attack
	ACTION_TYPE = initReq->action_type();

	if (ACTION_TYPE == 0)
	{
		DISTANCE = initReq->move_dist() * TILE_SIZE;
		MOVE_ANGLE = initReq->move_angle();

		ACTION_SIZE = 360 / MOVE_ANGLE + 1;

		// 시작점은 3시 방향으로 한다. ( DISTANCE )
		Position standardPos = Position(DISTANCE, 0);

		int totalAngle = 0;

		while (totalAngle < 360) {
			MovePosition.push_back(getCirclePosFromPosByDegree(Positions::Origin, standardPos, totalAngle));
			totalAngle += MOVE_ANGLE;
		}
	}

	STEP_FRAME = initReq->frames_per_step();
}

bool RLAvoidReavers::initializeAndValidate()
{
	return true;
}

void RLAvoidReavers::makeInitMessage(::google::protobuf::Message *initMessage)
{
	InitRes *message = (InitRes *)initMessage;

	if (ACTION_TYPE == 0)
		message->set_num_action_space(ACTION_SIZE);
	else
		message->set_num_action_space(2);

	setTypeInfo(message->add_unit_type_map(), Terran_Dropship);
	setTypeInfo(message->add_unit_type_map(), Protoss_Reaver);
}


//// 리셋
void RLAvoidReavers::reset(bool isFirstResetCall)
{
	if (agent && agent->getID() > 9000)
		restartGame();

	agent = nullptr;
	lastTargetPos = Positions::Origin;
	invalidAction = false;
}

bool RLAvoidReavers::isResetFinished()
{
	if (agent == nullptr && INFO.getUnits(Terran_Dropship, S).size() && INFO.getUnits(Protoss_Reaver, E).size()) {
		agent = INFO.getUnits(Terran_Dropship, S)[0]->unit();

		if (agent->getPosition().x < 50 && agent->getPosition().y < 50)
			return true;
	}

	return false;
}


//// Action
void RLAvoidReavers::step(::google::protobuf::Message *stepReqMsg)
{
	UnitInfo *s = INFO.getUnitInfo(agent, S);

	if (!s)
		return;

	Action act = ((StepReq *)stepReqMsg)->action(0);

	startedWithOverlapped = false;
	isOverlapped = false;

	UnitInfo *closest = INFO.getClosestTypeUnit(E, agent->getPosition(), Protoss_Reaver, 100);

	if (closest) {
		if (agent->getDistance(closest->unit()) <= hitThreshHold)
			startedWithOverlapped = true;
	}

	Position targetPos = Positions::None;

	if (ACTION_TYPE == 0)
	{

		lastAction = act.action_num();

		if (act.action_num() == ACTION_SIZE - 1) {
			agent->stop();
			lastTargetPos = agent->getPosition();
		}
		else {
			Position next = getValidPosition(agent->getPosition(), MovePosition[act.action_num()]);
			agent->move(next);
			lastTargetPos = next;
		}
	}
	else if (ACTION_TYPE == 1)
	{
		targetPos = { s->pos().x + act.pos_x(), s->pos().y + act.pos_y() };

		if (act.action_num() == 0)
			CommandUtil::move(s->unit(), targetPos);
		else if (act.action_num() == 1)
			CommandUtil::attackMove(s->unit(), targetPos);
	}
	else if (ACTION_TYPE == 2)
	{
		int angle = (int)(act.angle() * 360);
		targetPos = getCirclePosFromPosByDegree(s->pos(), s->pos() + Position(0, -act.radius() * TILE_SIZE), angle);

		Position next = getValidPosition(agent->getPosition(), targetPos);
		agent->move(next);
		lastTargetPos = next;
	}
	else {}

	//cout << "[next] lastAction" << lastAction << "," << next << "," << (TilePosition)next << endl;
	return;
}

bool RLAvoidReavers::isActionFinished()
{
	if (isDone()) {
		//cout << "Done 이라 끝" << endl;
		return true;
	}

	if (isLogicalDone()) {
		agent->move(GOAL_POS);
		return false;
	}

	if (!startedWithOverlapped && !isOverlapped) {
		UnitInfo *closest = INFO.getClosestTypeUnit(E, agent->getPosition(), Protoss_Reaver, 100);

		if (closest) {
			if (agent->getDistance(closest->unit()) <= hitThreshHold) {
				isOverlapped = true;
				bw->drawCircleMap(agent->getPosition(), 16, Colors::Red, true);
			}
		}
	}

	if (STEP_FRAME != -1)
		return (TIME - startFrame) % STEP_FRAME == 0;

	if (lastAction == (ACTION_SIZE - 1)) {
		return (TIME - startFrame) % 24 == 0;
	}

	if (lastTargetPos.getDistance(agent->getPosition()) <= 32) {
		return true;
	}

	return false;
}

void RLAvoidReavers::getObservation(::google::protobuf::Message *stateMsg)
{
	// Action이 끝나고 초기화
	invalidAction = false;

	State *stateMessage = (State *)stateMsg;

	if (isDone()) {
		UInfo *my_info = stateMessage->add_my_unit();

		my_info->set_unit_type(Terran_Dropship.getName());
		my_info->set_pos_x(GOAL_POS.x);
		my_info->set_pos_y(GOAL_POS.y);

		return;
	}

	setUInfo(stateMessage->add_my_unit(), agent);

	uList Reavers = INFO.getUnits(Protoss_Reaver, E);

	sort(Reavers.begin(), Reavers.end(), [](UnitInfo * a, UnitInfo * b) {
		if (a->pos().x < b->pos().x)
			return true;
		else if (a->pos().x > b->pos().x)
			return false;
		else if (a->pos().y < b->pos().y)
			return true;
		else
			return false;
	});

	for (auto rv : Reavers)
		setUInfo(stateMessage->add_en_unit(), rv);
}


//// Done, Reward
bool RLAvoidReavers::isDone()
{
	return agent && !agent->exists();
}

bool BWML::RLAvoidReavers::isLogicalDone()
{
	if (agent && agent->exists()) {
		Position agentPos = agent->getPosition();

		// 싱크가 잘 안되서 이미 골에 도착했는데도 에이전트가 살아 있을 수도 있다.
		if (agentPos.x > 256 && agentPos.y > 256)
			return true;
	}

	return false;
}

float RLAvoidReavers::getReward() {
	if (isDone())
		return WIN_REWARD;

	if (isOverlapped) {
		bw->drawCircleMap(agent->getPosition(), 16, Colors::Red, true);
		return -1;
	}

	if (startedWithOverlapped) {
		UnitInfo *closest = INFO.getClosestTypeUnit(E, agent->getPosition(), Protoss_Reaver, 100);

		// Step 이동 후에도 리버와 겹쳐있으면 마이너스
		if (closest)
			if (agent->getDistance(closest->unit()) <= hitThreshHold) {
				bw->drawCircleMap(agent->getPosition(), 16, Colors::Red, true);
				return -1;
			}
	}

	if (invalidAction)
		return -0.1f;

	return 0;
}


// 렌더 그 외
void RLAvoidReavers::render()
{

	if (!agent)
		return;

	// Move Cam to Agent
	focus(agent->getPosition());

	// Draw next position's of agent
	bw->drawCircleMap(lastTargetPos, 5, Colors::Yellow, true);

	if (!INFO.getUnits(Terran_Dropship, S).size())
		return;

	bw->drawBoxMap(Position(agent->getLeft(), agent->getTop()),
				   Position(agent->getRight(), agent->getBottom()), Colors::Red, false);

	uList Observers = INFO.getUnits(Protoss_Reaver, E);

	for (auto ob : Observers)
	{
		bw->drawBoxMap(Position(ob->unit()->getLeft(), ob->unit()->getTop()),
					   Position(ob->unit()->getRight(), ob->unit()->getBottom()), Colors::Red, false);
		bw->drawLineMap(ob->pos(), ob->pos() + Position((int)(ob->unit()->getVelocityX() * 10), (int)(ob->unit()->getVelocityY() * 10)), Colors::Cyan);
	}

}

Position RLAvoidReavers::getValidPosition(Position from, Position direction) {

	Position next = from + direction;
	int cnt = 1;

	int paddingDown = Terran_Dropship.dimensionDown() + 16;
	int paddingLeft = Terran_Dropship.dimensionLeft() + 16;
	int paddingRight = Terran_Dropship.dimensionRight() + 16;
	int paddingUp = Terran_Dropship.dimensionUp() + 16;

	// 오른쪽으로 벗어난 경우
	if (next.x > map_size - paddingRight) {
		if (next.x != from.x)
			next.y = from.y + (from.x - (map_size - paddingRight)) * (from.y - next.y) / (next.x - from.x);

		next.x = map_size - paddingRight;
	}

	// 왼쪽으로 벗어난 경우
	if (next.x < paddingLeft) {
		if (next.x != from.x)
			next.y = from.y + (from.x - paddingLeft) * (from.y - next.y) / (next.x - from.x);

		next.x = paddingLeft;
	}

	// 아래쪽으로 벗어난 경우
	if (next.y > map_size - paddingDown) {
		if (next.y != from.y)
			next.x = from.x + (from.x - next.x) * (from.y - map_size - paddingDown) / (next.y - from.y);

		next.y = map_size - paddingDown;
	}

	// 위쪽으로 벗어난 경우
	if (next.y < paddingUp) {
		if (next.y != from.y)
			next.x = from.x + (from.x - next.x) * (from.y - paddingUp) / (next.y - from.y);

		next.y = paddingUp;
	}

	if (from.getDistance(next) < 10)
		invalidAction = true;

	return next;
}

void RLAvoidReavers::onUnitDestroy(Unit unit)
{
}
