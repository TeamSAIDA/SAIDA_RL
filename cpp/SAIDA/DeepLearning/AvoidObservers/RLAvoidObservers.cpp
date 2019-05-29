/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "RLAvoidObservers.h"
#include "../../UXManager.h"

using namespace BWML;
using namespace MyBot;

RLAvoidObservers &RLAvoidObservers::Instance(string shmName) {
	static RLAvoidObservers instance(shmName);
	return instance;
}

// Init
void RLAvoidObservers::init(::google::protobuf::Message *message)
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

		ACTION_SIZE = 360 / MOVE_ANGLE;

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

bool RLAvoidObservers::initializeAndValidate()
{
	return true;
}

void RLAvoidObservers::makeInitMessage(::google::protobuf::Message *initMessage)
{
	InitRes *message = (InitRes *)initMessage;

	if (ACTION_TYPE == 0)
		message->set_num_action_space(ACTION_SIZE + 1);
	else
		message->set_num_action_space(2);

	TypeInfo *myUnitType = message->mutable_my_unit_type();

	UnitType myType = Zerg_Scourge;

	myUnitType->set_hp_max(myType.maxHitPoints());
	myUnitType->set_shield_max(myType.maxShields());
	myUnitType->set_energy_max(myType.maxEnergy());
	myUnitType->set_armor(myType.armor());
	myUnitType->set_cooldown_max(myType.groundWeapon().damageCooldown());
	myUnitType->set_acceleration(myType.acceleration());
	myUnitType->set_top_speed(myType.topSpeed());
	myUnitType->set_damage_amount(myType.groundWeapon().damageAmount());
	myUnitType->set_damage_factor(myType.groundWeapon().damageFactor());
	myUnitType->set_weapon_range(myType.groundWeapon().maxRange());
	myUnitType->set_sight_range(myType.sightRange());
	myUnitType->set_seek_range(myType.seekRange());

	TypeInfo *enUnitType = message->mutable_en_unit_type();

	myType = Protoss_Observer;

	enUnitType->set_hp_max(myType.maxHitPoints());
	enUnitType->set_shield_max(myType.maxShields());
	enUnitType->set_energy_max(myType.maxEnergy());
	enUnitType->set_armor(myType.armor());
	enUnitType->set_cooldown_max(myType.groundWeapon().damageCooldown());
	enUnitType->set_acceleration(myType.acceleration());
	enUnitType->set_top_speed(myType.topSpeed());
	enUnitType->set_damage_amount(myType.groundWeapon().damageAmount());
	enUnitType->set_damage_factor(myType.groundWeapon().damageFactor());
	enUnitType->set_weapon_range(myType.groundWeapon().maxRange());
	enUnitType->set_sight_range(myType.sightRange());
	enUnitType->set_seek_range(myType.seekRange());
}


// Reset
void RLAvoidObservers::reset(bool isFirstResetCall)
{
	if (agent && agent->getID() > 9000)
		restartGame();

	agent = nullptr;
	lastTargetPos = Positions::Origin;
}

bool RLAvoidObservers::isResetFinished()
{
	if (agent == nullptr && INFO.getUnits(Zerg_Scourge, S).size() == 1 &&
			INFO.getUnits(Protoss_Observer, E).size() >= 1) {
		agent = INFO.getUnits(Zerg_Scourge, S)[0]->unit();
		return true;
	}

	return false;
}


// Action
void RLAvoidObservers::step(::google::protobuf::Message *stepReqMsg)
{
	UnitInfo *s = INFO.getUnitInfo(agent, S);

	if (!s)
		return;

	Action act = ((StepReq *)stepReqMsg)->action(0);

	Position targetPos = Positions::None;

	if (ACTION_TYPE == 0)
	{
		vector<int> myAlts = getNearObstacle(s, ACTION_SIZE);

		targetPos = s->pos() + MovePosition[act.action_num()];

		if (myAlts[act.action_num()] < DISTANCE)
			targetPos = getDirectionDistancePosition(s->pos(), targetPos, myAlts[act.action_num()]);

		CommandUtil::rightClick(s->unit(), targetPos);
		commandPosition = s->pos();
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
	}
	else {}

	CommandUtil::move(agent, targetPos);
	lastTargetPos = targetPos;
	commandTime = TIME;
	//cout << "[next] lastAction" << lastAction << "," << next << "," << (TilePosition)next << endl;
	return;
}

bool RLAvoidObservers::isActionFinished()
{
	if (isDone())
		return true;

	if (isLogicalDone()) {
		// 위로 이동..
		CommandUtil::move(agent, Position(agent->getPosition().x, 0));
		return false;
	}

	if (STEP_FRAME != -1)
		return (TIME - startFrame) % STEP_FRAME == 0;

	if (ACTION_TYPE == 0 && lastAction == ACTION_SIZE) // Command Stop을 의미함. 이때는 6 프레임 후에 Finish
		return commandTime + 6 < TIME;

	// Attack Command 였다면
	double actionVectorX, actionVectorY;
	actionVectorX = (double)(lastTargetPos - agent->getPosition()).x / lastTargetPos.getDistance(agent->getPosition());
	actionVectorY = (double)(lastTargetPos - agent->getPosition()).y / lastTargetPos.getDistance(agent->getPosition());

	int thresholdGap = TILE_SIZE;
	Position thresholdActionMove = { (int)(actionVectorX * thresholdGap), (int)(actionVectorY * thresholdGap) };

	Position newPos = agent->getPosition() - commandPosition;

	bool overX = false, overY = false;

	if (thresholdActionMove.x <= 0 && newPos.x <= thresholdActionMove.x ||
			thresholdActionMove.x >= 0 && newPos.x >= thresholdActionMove.x)
		overX = true;

	if (thresholdActionMove.y <= 0 && newPos.y <= thresholdActionMove.y ||
			thresholdActionMove.y >= 0 && newPos.y >= thresholdActionMove.y)
		overY = true;

	if (overX && overY)
	{
		return true;
	}

	return false;
}

void RLAvoidObservers::getObservation(::google::protobuf::Message *stateMsg)
{
	State *stateMessage = (State *)stateMsg;

	UInfo *my_info = stateMessage->add_my_unit();

	if (isDone()) {
		my_info->set_pos_x(destroyPos.x);
		my_info->set_pos_y(destroyPos.y);
		my_info->set_angle(0.0);
		my_info->set_velocity_x(0.0);
		my_info->set_velocity_y(0.0);

		return;
	}

	UnitInfo *s = INFO.getUnits(Zerg_Scourge, S)[0];

	my_info->set_pos_x(s->pos().x);
	my_info->set_pos_y(s->pos().y);
	my_info->set_angle(s->unit()->getAngle());
	my_info->set_velocity_x(s->unit()->getVelocityX());
	my_info->set_velocity_y(s->unit()->getVelocityY());
	my_info->set_accelerating(s->unit()->isAccelerating());


	Position TL = (s->pos() - Position(OBSERVABLE_TILE_SPACE_RADIUS * TILE_SIZE, OBSERVABLE_TILE_SPACE_RADIUS * TILE_SIZE)).makeValid();
	Position BR = (s->pos() + Position(OBSERVABLE_TILE_SPACE_RADIUS * TILE_SIZE, OBSERVABLE_TILE_SPACE_RADIUS * TILE_SIZE)).makeValid();

	uList Observers = INFO.getTypeUnitsInRectangle(Protoss_Observer, E, TL, BR);
	//int obsIdx = 0;
	UListSet obs;

	for (auto ob : Observers) {
		obs.add(ob);
	}

	Observers = obs.getSortedUnitList(agent->getPosition());

	for (auto ob : Observers)
	{
		UInfo *ob_info = stateMessage->add_en_unit();

		ob_info->set_pos_x(ob->pos().x);
		ob_info->set_pos_y(ob->pos().y);
		ob_info->set_angle(ob->unit()->getAngle());
		ob_info->set_velocity_x(ob->unit()->getVelocityX());
		ob_info->set_velocity_y(ob->unit()->getVelocityY());
		ob_info->set_accelerating(ob->unit()->isAccelerating());
	}
}


// Done, Reward
bool RLAvoidObservers::isDone()
{
	return agent && !agent->exists();
}

bool RLAvoidObservers::isLogicalDone() {
	if (agent && agent->exists()) {
		Position agentPos = agent->getPosition();

		// 싱크가 잘 안되서 이미 골에 도착했는데도 에이전트가 살아 있을 수도 있다.
		if (agentPos.y > 0 && agentPos.y <= 64) {
			cout << "My agent observer reached the destination. This episode is succeed." << endl;
			return true;
		}
	}

	return false;
}

float RLAvoidObservers::getReward() {
	float reward = .0;

	// Basic Reward

	// 1. 생존
	if (agent && agent->exists())
		reward = 0.5;
	// 2. 중간에 죽음
	else {
		if (destroyPos.y <= 64)
			reward = 5.;
		else
			reward = -5.;
	}

	return reward;
}


// Etc
void RLAvoidObservers::render()
{
	if (lastAction == -1)
		return;

	if (agent)
	{
		focus(agent->getPosition());

		Position pos = agent->getPosition();

		Position st(-OBSERVABLE_TILE_SPACE_RADIUS * TILE_SIZE, -OBSERVABLE_TILE_SPACE_RADIUS * TILE_SIZE);


		for (int i = 0; i < OBSERVABLE_TILE_SPACE_RADIUS * 2 + 1; i++) {
			bw->drawLineMap(pos + st + Position(0, i * TILE_SIZE), pos + st + Position(OBSERVABLE_TILE_SPACE_RADIUS * 2 * TILE_SIZE, i * TILE_SIZE), Colors::Cyan);
			bw->drawLineMap(pos + st + Position(i * TILE_SIZE, 0), pos + st + Position(i * TILE_SIZE, OBSERVABLE_TILE_SPACE_RADIUS * 2 * TILE_SIZE), Colors::Cyan);
		}

		Position TL = (pos - Position(OBSERVABLE_TILE_SPACE_RADIUS * TILE_SIZE, OBSERVABLE_TILE_SPACE_RADIUS * TILE_SIZE)).makeValid();
		Position BR = (pos + Position(OBSERVABLE_TILE_SPACE_RADIUS * TILE_SIZE, OBSERVABLE_TILE_SPACE_RADIUS * TILE_SIZE)).makeValid();
		uList Observers = INFO.getTypeUnitsInRectangle(Protoss_Observer, E, TL, BR);

		for (auto ob : Observers)
		{
			if (ob->pos().isValid()) {
				bw->drawTextMap(ob->pos() + Position(0, 10), "(%.2f, %.2f)", ob->unit()->getVelocityX(), ob->unit()->getVelocityY());
				bw->drawTextMap(ob->pos() + Position(0, 20), "(%d, %d)", ob->pos().x, ob->pos().y);
				bw->drawCircleMap(ob->vPos(), 5, Colors::Red);
			}
		}
	}

	if (agent)
		bw->drawCircleMap(agent->getTargetPosition(), 6, Colors::Yellow, true);

}

void RLAvoidObservers::onUnitDestroy(Unit unit)
{
	if (unit->getType() == Zerg_Scourge)
		destroyPos = unit->getPosition();
}
