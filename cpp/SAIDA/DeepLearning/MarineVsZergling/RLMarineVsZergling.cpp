/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "RLMarineVsZergling.h"
#include "../../UXManager.h"

using namespace BWML;
using namespace MyBot;

RLMarineVsZergling &RLMarineVsZergling::Instance(string shmName) {
	static RLMarineVsZergling instance(shmName);
	return instance;
}

// Init
void RLMarineVsZergling::init(::google::protobuf::Message *message)
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

	MY_UNIT_COUNT = 3;
	ENEMY_UNIT_COUNT = 4;
	bw->setVision(E, true);
}

bool RLMarineVsZergling::initializeAndValidate()
{
	if (getIsResetting() && isResetFinished()) {
		for (auto v : INFO.getUnits(Terran_Marine, S)) {
			idOrderedList.push_back(v->id());
		}

		for (auto z : INFO.getUnits(Zerg_Zergling, E)) {
			enemyIdOrderedList.push_back(z->id());
		}

		cout << "한번만 불러줘~~ ^^!!" << std::endl;
	}

	return !(INFO.getUnits(Terran_Marine, S).empty() && INFO.getUnits(Zerg_Zergling, E).empty());
}

void RLMarineVsZergling::makeInitMessage(::google::protobuf::Message *initMessage)
{
	InitRes *message = (InitRes *)initMessage;

	if (ACTION_TYPE == 0)
		message->set_num_action_space(ACTION_SIZE + 1);
	else
		message->set_num_action_space(2);

	TypeInfo *marine = message->mutable_my_unit_type();

	UnitType myType = Terran_Marine;

	marine->set_hp_max(myType.maxHitPoints());
	marine->set_shield_max(myType.maxShields());
	marine->set_energy_max(myType.maxEnergy());
	marine->set_armor(myType.armor());
	marine->set_cooldown_max(myType.groundWeapon().damageCooldown());
	marine->set_acceleration(myType.acceleration());
	marine->set_top_speed(myType.topSpeed());
	marine->set_damage_amount(myType.groundWeapon().damageAmount());
	marine->set_damage_factor(myType.groundWeapon().damageFactor());
	marine->set_weapon_range(myType.groundWeapon().maxRange());
	marine->set_sight_range(myType.sightRange());
	marine->set_seek_range(myType.seekRange());

	TypeInfo *zerlging = message->mutable_en_unit_type();

	myType = Zerg_Zergling;

	zerlging->set_hp_max(myType.maxHitPoints());
	zerlging->set_shield_max(myType.maxShields());
	zerlging->set_energy_max(myType.maxEnergy());
	zerlging->set_armor(myType.armor());
	zerlging->set_cooldown_max(myType.groundWeapon().damageCooldown());
	zerlging->set_acceleration(myType.acceleration());
	zerlging->set_top_speed(myType.topSpeed());
	zerlging->set_damage_amount(myType.groundWeapon().damageAmount());
	zerlging->set_damage_factor(myType.groundWeapon().damageFactor());
	zerlging->set_weapon_range(myType.groundWeapon().maxRange());
	zerlging->set_sight_range(myType.sightRange());
	zerlging->set_seek_range(myType.seekRange());

	for (int i = 0; i < theMap.WalkSize().x; i++)
		for (int j = 0; j < theMap.WalkSize().y; j++)
			message->add_iswalkable(bw->isWalkable(WalkPosition(i, j)));
}


// Reset
void RLMarineVsZergling::reset(bool isFirstResetCall)
{
	if (!idOrderedList.empty() && idOrderedList.at(0) > 9000)
		restartGame();

	INFO.clearUnitNBuilding();

	idOrderedList.clear();
	enemyIdOrderedList.clear();

	for (auto unit : bw->getAllUnits()) {
		if (unit->exists())
			INFO.onUnitComplete(unit);
	}

	myKillCount = 0;
	enemyKillCount = 0;
}

bool RLMarineVsZergling::isResetFinished()
{
	return INFO.getUnits(Terran_Marine, S).size() == MY_UNIT_COUNT && INFO.getUnits(Zerg_Zergling, E).size() == ENEMY_UNIT_COUNT;
}


// Action
void RLMarineVsZergling::step(::google::protobuf::Message *stepReqMsg)
{
	if (HUMAN_MODE)
		return;

	uList agents = INFO.getUnits(Terran_Marine, S);

	for (int i = 0; i < MY_UNIT_COUNT; i++) {
		Action act = ((StepReq *)stepReqMsg)->action(i);

		for (auto s : agents) {
			if (s->id() == idOrderedList[i]) {
				Position targetPos = Positions::None;

				if (ACTION_TYPE == 0)
				{
					if (act.action_num() == ACTION_SIZE)
					{
						UnitInfo *weakUnit = getGroundWeakTargetInRange(s);

						if (weakUnit)
							CommandUtil::attackUnit(s->unit(), weakUnit->unit());
						else
						{
							UnitInfo *closest = INFO.getClosestTypeUnit(E, s->pos(), Protoss_Zealot);

							if (closest)
								CommandUtil::attackUnit(s->unit(), closest->unit());
						}
					}
					else
					{
						vector<int> myAlts = getNearObstacle(s, ACTION_SIZE);

						targetPos = s->pos() + MovePosition[act.action_num()];

						if (myAlts[act.action_num()] < DISTANCE)
							targetPos = getDirectionDistancePosition(s->pos(), targetPos, myAlts[act.action_num()]);

						CommandUtil::rightClick(s->unit(), targetPos);
						commandPosition = s->pos();
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
				}
				else {}
			}
		}
	}

	commandTime = TIME;

	return;
}

bool RLMarineVsZergling::isActionFinished()
{
	if (HUMAN_MODE)
		return true;

	if (isDone())
		return true;

	return (TIME - startFrame) % STEP_FRAME == 0;
}

void RLMarineVsZergling::getObservation(::google::protobuf::Message *stateMsg)
{
	State *stateMessage = (State *)stateMsg;

	uList agents = INFO.getUnits(Terran_Marine, S);

	for (auto id : idOrderedList) {

		bool isAlive = false;

		for (auto m : agents) {
			if (id == m->id()) {
				// 살아있는거 정보 넣기.
				UInfo *m_info = stateMessage->add_my_unit();

				m_info->set_accelerating(m->unit()->isAccelerating());
				m_info->set_angle(m->unit()->getAngle());
				m_info->set_attacking(m->unit()->isAttacking());
				m_info->set_cooldown(m->unit()->getGroundWeaponCooldown());
				m_info->set_hp(m->unit()->getHitPoints());
				m_info->set_energy(m->unit()->getEnergy());
				m_info->set_shield(m->unit()->getShields());
				m_info->set_pos_x(m->unit()->getPosition().x);
				m_info->set_pos_y(m->unit()->getPosition().y);
				m_info->set_velocity_x(m->unit()->getVelocityX());
				m_info->set_velocity_y(m->unit()->getVelocityY());

				if (ACTION_TYPE == 0)
				{
					for (auto alt : getNearObstacle(m, ACTION_SIZE))
					{
						TerrainInfo *posInfo = m_info->add_pos_info();

						posInfo->set_nearest_obstacle_dist(alt);
					}
				}

				isAlive = true;
				break;
			}
		}

		// 빈거 넣어주기.
		if (!isAlive) {
			stateMessage->add_my_unit();
		}
	}

	uList targets = INFO.getUnits(Zerg_Zergling, E);

	for (auto id : enemyIdOrderedList) {

		bool isAlive = false;

		for (auto z : targets) {
			if (id == z->id()) {
				// 살아있는거 정보 넣기.
				UInfo *z_info = stateMessage->add_en_unit();

				z_info->set_accelerating(z->unit()->isAccelerating());
				z_info->set_angle(z->unit()->getAngle());
				z_info->set_attacking(z->unit()->isAttacking());
				z_info->set_cooldown(z->unit()->getGroundWeaponCooldown());
				z_info->set_hp(z->unit()->getHitPoints());
				z_info->set_energy(z->unit()->getEnergy());
				z_info->set_shield(z->unit()->getShields());
				z_info->set_pos_x(z->unit()->getPosition().x);
				z_info->set_pos_y(z->unit()->getPosition().y);
				z_info->set_velocity_x(z->unit()->getVelocityX());
				z_info->set_velocity_y(z->unit()->getVelocityY());

				isAlive = true;
				break;
			}
		}

		// 빈거 넣어주기.
		if (!isAlive)
			stateMessage->add_en_unit();
	}
}

// Done, Reward
bool RLMarineVsZergling::isDone()
{
	return isWin() || isDefeat();
}

float RLMarineVsZergling::getReward() {
	float reward = 0;

	if (isWin())
		reward = 1.0;
	else if (isDefeat())
		reward = -1.0;
	else {
		reward = 0;
	}

	return reward;
}


// Etc
bool RLMarineVsZergling::isWin() {
	return INFO.getDestroyedCount(Zerg_Zergling, E) == ENEMY_UNIT_COUNT;
}

bool RLMarineVsZergling::isDefeat() {
	return INFO.getDestroyedCount(Terran_Marine, S) == MY_UNIT_COUNT;
}

bool RLMarineVsZergling::isTimeout() {
	return false;
}

void RLMarineVsZergling::render()
{
	uList self = INFO.getUnits(Terran_Marine, S);

	if (!self.empty()) {
		Position target;

		for (auto s : self)
			target += s->pos();

		focus(target / self.size());
	}
	else
		focus(theMap.Center());
}

void RLMarineVsZergling::onUnitDestroy(Unit unit) {
	if (!isDone() && !getIsResetting()) {
		if (unit->getPlayer() == S)
			myKillCount += unit->getKillCount();
		else
			enemyKillCount += unit->getKillCount();
	}
}