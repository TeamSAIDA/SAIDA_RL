/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "RLAvoidZerglings.h"
#include "../../UXManager.h"

using namespace BWML;
using namespace MyBot;

RLAvoidZerglings &RLAvoidZerglings::Instance(string shmName) {
	static RLAvoidZerglings instance(shmName);
	return instance;
}

//// INIT 관련
void RLAvoidZerglings::init(::google::protobuf::Message *message)
{
	InitReq *initReq = (InitReq *)message;

	int map_version = initReq->version();

	// ACTION TYPE : 0 -> Action Number
	// ACTION TYPE : 1 -> Position X, Y Action Number
	// ACTION TYPE : 2 -> Angle, Radius, Action Number
	// ACTION NUMBER : 0 - Move, 1 - Attack
	ACTION_TYPE = initReq->action_type();

	if (ACTION_TYPE == 0 || ACTION_TYPE == 3)
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
	RADIAN_UNIT = 2 * M_PI / ACTION_SIZE;
	lastGas = 0;
}

bool RLAvoidZerglings::initializeAndValidate()
{
	return true;
}

void RLAvoidZerglings::makeInitMessage(::google::protobuf::Message *initMessage)
{
	InitRes *message = (InitRes *)initMessage;

	if (ACTION_TYPE == 0 || ACTION_TYPE == 3)
		message->set_num_action_space(ACTION_SIZE);
	else
		message->set_num_action_space(2);

	TypeInfo *myUnitType = message->mutable_my_unit_type();

	UnitType myType = agent_type;

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

	myType = Zerg_Zergling;

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

//// 리셋
void RLAvoidZerglings::reset(bool isFirstResetCall)
{
	if (agent && agent->getID() > 9500) {
		lastGas = 0;

		if (bw->isMultiplayer())
			leaveGame();
		else
			restartGame();
	}

	agent = nullptr;
	lastTargetPos = Positions::Origin;
	lastReward = 0;
}

bool RLAvoidZerglings::isResetFinished()
{

	if (agent == nullptr && INFO.getUnits(agent_type, S).size()) {
		agent = INFO.getUnits(agent_type, S)[0]->unit();
		return true;
	}


	return false;
}

//// Action
void RLAvoidZerglings::step(::google::protobuf::Message *stepReqMsg)
{
	UnitInfo *s = INFO.getUnitInfo(agent, S);

	if (!s)
		return;

	Action act = ((StepReq *)stepReqMsg)->action(0);

	Position targetPos = Positions::None;

	if (ACTION_TYPE == 0)
	{

		lastAction = act.action_num();

		Position next = agent->getPosition() + MovePosition[act.action_num()];
		agent->move(next);
		lastTargetPos = next;

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
	else {
		// Rule

		direction = direction * goWithoutDamage(s->unit(), Position(0, 0), direction);

	}

	//cout << "[next] lastAction" << lastAction << "," << next << "," << (TilePosition)next << endl;
	return;
}

bool RLAvoidZerglings::isActionFinished()
{

	if (isDone()) {
		return true;
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

void RLAvoidZerglings::getObservation(::google::protobuf::Message *stateMsg)
{

	State *observation = (State *)stateMsg;

	// SCV Information
	UInfo *me = observation->add_my_unit();
	me->set_accelerating(agent->isAccelerating());
	me->set_braking(agent->isBraking());
	me->set_is_attack_frame(agent->isAttackFrame());
	me->set_angle(agent->getAngle());
	me->set_attacking(agent->isAttacking());
	me->set_cooldown(agent->getGroundWeaponCooldown());
	me->set_hp(agent->getHitPoints());
	me->set_energy(agent->getEnergy());
	me->set_shield(agent->getShields());
	me->set_pos_x(agent->getPosition().x);
	me->set_pos_y(agent->getPosition().y);
	me->set_velocity_x(agent->getVelocityX());
	me->set_velocity_y(agent->getVelocityY());

	if (isDone()) {
		return;
	}

	UnitInfo *u = INFO.getUnitInfo(agent, S);

	if (ACTION_TYPE == 0) {

		// Distance from unwalkable tiles of each direction
		for (auto alt : getNearObstacle(u, ACTION_SIZE, true)) {
			TerrainInfo *posInfo = me->add_pos_info();
			posInfo->set_nearest_obstacle_dist(alt);
		}

		// <radian, distance>
		for (auto e : getRadianAndDistanceFromEnemy(u, ACTION_SIZE)) {
			TerrainInfo *posInfo = me->add_pos_info();
			int ind = (int)(e.first / RADIAN_UNIT);
			posInfo->set_udi_int_1(ind); // index which indicates
			posInfo->set_udi_double_1(e.first); // radian
			posInfo->set_udi_double_2(e.second); // distance

			TerrainInfo *posInfo2 = me->add_pos_info();
			int prev_ind = ind - 1;
			prev_ind = prev_ind >= 0 ? prev_ind : ACTION_SIZE - 1;
			posInfo2->set_udi_int_1(prev_ind); // index which indicates
			posInfo2->set_udi_double_1(e.first); // radian
			posInfo2->set_udi_double_2(e.second * 1.5); // distance

			TerrainInfo *posInfo3 = me->add_pos_info();
			int next_ind = ind + 1;
			next_ind = next_ind < ACTION_SIZE ? next_ind : 0;
			posInfo3->set_udi_int_1(next_ind); // last index
			posInfo3->set_udi_double_1(e.first); // radian
			posInfo3->set_udi_double_2(e.second * 1.5); // distance

		}

	}

	// Zergling Information
	uList zerglings = INFO.getUnits(Zerg_Zergling, E);

	for (auto z : zerglings) {
		UInfo *e = observation->add_en_unit();
		e->set_accelerating(z->unit()->isAccelerating());
		e->set_braking(z->unit()->isBraking());
		e->set_is_attack_frame(z->unit()->isAttackFrame());
		e->set_angle(z->unit()->getAngle());
		e->set_attacking(z->unit()->isAttacking());
		e->set_cooldown(z->unit()->getGroundWeaponCooldown());
		e->set_hp(z->unit()->getHitPoints());
		e->set_energy(z->unit()->getEnergy());
		e->set_shield(z->unit()->getShields());
		e->set_pos_x(z->unit()->getPosition().x);
		e->set_pos_y(z->unit()->getPosition().y);
		e->set_velocity_x(z->unit()->getVelocityX());
		e->set_velocity_y(z->unit()->getVelocityY());
	}

}

//// Done, Reward
bool RLAvoidZerglings::isDone()
{
	if (agent && !agent->exists())
		return true;

	return false;
}

float RLAvoidZerglings::getReward() {

	float reward = 0;

	// Basic Reward
	// 1. 생존
	if (agent && agent->exists()) {
		reward = -1;
	}

	if (S->gas() > lastGas) {
		reward = 1;
		lastGas = S->gas();
	}

	lastReward = reward;

	return reward;
}


// 렌더 그 외
void RLAvoidZerglings::render()
{

	if (!agent)
		return;

	// Move Cam to Agent
	focus(agent->getPosition());

	// Draw next position's of agent
	bw->drawCircleMap(lastTargetPos, 5, Colors::Yellow, true);

	UnitInfo *u = INFO.getUnitInfo(agent, S);

	if (u == nullptr)
		return;

	vector<pair<double, double>> v = getRadianAndDistanceFromEnemy(u, ACTION_SIZE); // radian, distance
	vector<double> weight;

	for (int i = 0; i < ACTION_SIZE; i++)
		weight.push_back(0);

	for (auto vv : v) {

		int ind = (int)(vv.first / RADIAN_UNIT);
		double dist = vv.second / 32;
		double dist_weight = 0;

		if (dist > 1) {
			dist_weight = 1 / (dist * dist);
		}
		else {
			dist_weight = dist;
		}

		weight.at(ind) += dist_weight;

		int prev_ind = ind - 1;
		prev_ind = prev_ind >= 0 ? prev_ind : ACTION_SIZE - 1;
		weight.at(prev_ind) += dist_weight * 0.8;

		int next_ind = ind + 1;
		next_ind = next_ind < ACTION_SIZE ? next_ind : 0;
		weight.at(next_ind) += dist_weight * 0.8;

	}

	// 36방에 대해서 직선 그리기
	int i = 0;
	int color_weight = 0;

	//cout << "------------------" << endl;

	for (auto pos : MovePosition) {
		//cout << "weight.at(" << i << ") = " << weight.at(i) << endl;
		color_weight = (int)(255 - (255 * weight.at(i)));
		color_weight = color_weight > 255 ? 255 : color_weight;
		color_weight = color_weight < 0 ? 0 : color_weight;

		//bw->drawLineMap(agent->getPosition(), agent->getPosition() + pos * 10, Colors::White);
		bw->drawLineMap(agent->getPosition(), agent->getPosition() + pos * 10, Color(255, color_weight, color_weight));
		i++;
	}

}

void RLAvoidZerglings::onUnitDestroy(Unit unit)
{
	if (unit->getType() == agent_type)
		destroyPos = unit->getPosition();
}
