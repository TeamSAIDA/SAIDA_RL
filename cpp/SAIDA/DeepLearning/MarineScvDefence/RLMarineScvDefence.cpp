/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "RLMarineScvDefence.h"
#include "../../UXManager.h"

using namespace BWML;
using namespace MyBot;

RLMarineScvDefence &RLMarineScvDefence::Instance(string shmName) {
	static RLMarineScvDefence instance(shmName);
	return instance;
}

// Init
void RLMarineScvDefence::init(::google::protobuf::Message *message)
{
	InitReq *initReq = (InitReq *)message;

	int map_version = initReq->version();

	// ACTION TYPE : 0 -> Action Number (Discrete Action)
	// ACTION TYPE : 1 -> Position X, Y Action Number (Continuous Action)
	// ACTION TYPE : 2 -> Angle, Radius, Action Number (Continuous Action)
	// ACTION NUMBER : 0 - Move, 1 - Attack (Continuous Action)
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

	MY_UNIT_COUNT[0] = 6;
	MY_UNIT_COUNT[1] = 9;
	ENEMY_UNIT_COUNT = 12;

	bw->setVision(E, true);

	lastAction = ACTION_SIZE;

	myUnitType[0] = Terran_Marine;
	myUnitType[1] = Terran_SCV;
	enUnitType = Zerg_Zergling;
}

bool RLMarineScvDefence::initializeAndValidate()
{
	if (getIsResetting() && isResetFinished()) {
		for (auto m : INFO.getUnits(myUnitType[0], S)) {
			idOrderedMarineList.push_back(m->id());
		}

		for (auto s : INFO.getUnits(myUnitType[1], S)) {
			idOrderedScvList.push_back(s->id());
		}

		for (auto z : INFO.getUnits(enUnitType, E)) {
			enemyIdOrderedList.push_back(z->id());
		}

		cout << "한번만 불러줘~~ ^^!!" << std::endl;
	}

	return !(INFO.getUnits(myUnitType[0], S).empty() && INFO.getUnits(enUnitType, E).empty());
}

void RLMarineScvDefence::makeInitMessage(::google::protobuf::Message *initMessage)
{
	InitRes *message = (InitRes *)initMessage;

	if (ACTION_TYPE == 0)
		message->set_num_action_space(ACTION_SIZE + 1);
	else
		message->set_num_action_space(2);

	TypeInfo *mUnitType = message->mutable_my_marine_type();

	UnitType myType = myUnitType[0];

	mUnitType->set_hp_max(myType.maxHitPoints());
	mUnitType->set_shield_max(myType.maxShields());
	mUnitType->set_energy_max(myType.maxEnergy());
	mUnitType->set_armor(myType.armor());
	mUnitType->set_cooldown_max(myType.groundWeapon().damageCooldown());
	mUnitType->set_acceleration(myType.acceleration());
	mUnitType->set_top_speed(myType.topSpeed());
	mUnitType->set_damage_amount(myType.groundWeapon().damageAmount());
	mUnitType->set_damage_factor(myType.groundWeapon().damageFactor());
	mUnitType->set_weapon_range(myType.groundWeapon().maxRange());
	mUnitType->set_sight_range(myType.sightRange());
	mUnitType->set_seek_range(myType.seekRange());

	mUnitType = message->mutable_my_scv_type();

	myType = myUnitType[1];

	mUnitType->set_hp_max(myType.maxHitPoints());
	mUnitType->set_shield_max(myType.maxShields());
	mUnitType->set_energy_max(myType.maxEnergy());
	mUnitType->set_armor(myType.armor());
	mUnitType->set_cooldown_max(myType.groundWeapon().damageCooldown());
	mUnitType->set_acceleration(myType.acceleration());
	mUnitType->set_top_speed(myType.topSpeed());
	mUnitType->set_damage_amount(myType.groundWeapon().damageAmount());
	mUnitType->set_damage_factor(myType.groundWeapon().damageFactor());
	mUnitType->set_weapon_range(myType.groundWeapon().maxRange());
	mUnitType->set_sight_range(myType.sightRange());
	mUnitType->set_seek_range(myType.seekRange());

	TypeInfo *eUnitType = message->mutable_en_unit_type();

	myType = enUnitType;

	eUnitType->set_hp_max(myType.maxHitPoints());
	eUnitType->set_shield_max(myType.maxShields());
	eUnitType->set_energy_max(myType.maxEnergy());
	eUnitType->set_armor(myType.armor());
	eUnitType->set_cooldown_max(myType.groundWeapon().damageCooldown());
	eUnitType->set_acceleration(myType.acceleration());
	eUnitType->set_top_speed(myType.topSpeed());
	eUnitType->set_damage_amount(myType.groundWeapon().damageAmount());
	eUnitType->set_damage_factor(myType.groundWeapon().damageFactor());
	eUnitType->set_weapon_range(myType.groundWeapon().maxRange());
	eUnitType->set_sight_range(myType.sightRange());
	eUnitType->set_seek_range(myType.seekRange());

	for (int i = 0; i < theMap.WalkSize().x; i++)
		for (int j = 0; j < theMap.WalkSize().y; j++)
			message->add_iswalkable(bw->isWalkable(WalkPosition(i, j)));
}


// Reset
void RLMarineScvDefence::reset(bool isFirstResetCall)
{
	if (!idOrderedMarineList.empty() && idOrderedMarineList.at(0) > 9000)
		restartGame();

	INFO.clearUnitNBuilding();

	idOrderedMarineList.clear();
	idOrderedScvList.clear();
	enemyIdOrderedList.clear();

	for (auto unit : bw->getAllUnits()) {
		if (unit->exists())
			INFO.onUnitComplete(unit);
	}

	myKillCount = 0;
	enemyKillCount = 0;
}

bool RLMarineScvDefence::isResetFinished()
{
	return INFO.getUnits(myUnitType[0], S).size() == MY_UNIT_COUNT[0] && INFO.getUnits(enUnitType, E).size() == ENEMY_UNIT_COUNT;
}


// Action
void RLMarineScvDefence::step(::google::protobuf::Message *stepReqMsg)
{
	if (HUMAN_MODE)
	{
		return;
	}

	Position targetPos = Positions::None;

	int idx = 0;

	for (Action act : ((StepReq *)stepReqMsg)->action())
	{
		if (ACTION_TYPE == 0)
		{
			if (act.action_num() == ACTION_SIZE)
			{
				Position target = getAvgPosition(INFO.getUnits(enUnitType, E));

				for (auto u : INFO.getUnits(myUnitType[idx], S))
					CommandUtil::attackMove(u->unit(), target);
			}
			else
			{

				Position source = getAvgPosition(INFO.getUnits(myUnitType[idx], S));
				targetPos = source + MovePosition[act.action_num()];

				for (auto u : INFO.getUnits(myUnitType[idx], S))
					CommandUtil::move(u->unit(), targetPos);
			}
		}

		idx++;
	}

	commandTime = TIME;
	// 9 : do nothing...
	return;
}

bool RLMarineScvDefence::isActionFinished()
{
	if (HUMAN_MODE)
		return true;

	if (isDone())
		return true;

	if (INFO.getUnits(myUnitType[0], S).empty() || INFO.getUnits(enUnitType, E).empty()) {
		cout << TIME << " 좀 있으면 isdone 될거야!(" << INFO.getUnits(myUnitType[0], S).size() << ", " << INFO.getUnits(enUnitType, E).size()
			 << ", " << isWin() << ", " << isDefeat() << ")" << endl;
		return false;
	}

	return (TIME - startFrame) % STEP_FRAME == 0;
}

void RLMarineScvDefence::getObservation(::google::protobuf::Message *stateMsg)
{
	State *stateMessage = (State *)stateMsg;

	uList agents = INFO.getUnits(Terran_Marine, S);

	for (auto id : idOrderedMarineList) {

		bool isAlive = false;

		for (auto m : agents) {
			if (id == m->id()) {
				// 살아있는거 정보 넣기.
				UInfo *m_info = stateMessage->add_my_marine();

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
				m_info->set_is_attack_frame(m->unit()->isAttackFrame());
				m_info->set_braking(m->unit()->isBraking());

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
			stateMessage->add_my_marine();
		}
	}

	agents = INFO.getUnits(Terran_SCV, S);

	for (auto id : idOrderedScvList) {

		bool isAlive = false;

		for (auto m : agents) {
			if (id == m->id()) {
				// 살아있는거 정보 넣기.
				UInfo *m_info = stateMessage->add_my_scv();

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
				m_info->set_is_attack_frame(m->unit()->isAttackFrame());
				m_info->set_braking(m->unit()->isBraking());

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
			stateMessage->add_my_scv();
		}
	}

	uList targets = INFO.getUnits(enUnitType, E);

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
				z_info->set_is_attack_frame(z->unit()->isAttackFrame());
				z_info->set_braking(z->unit()->isBraking());

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
bool RLMarineScvDefence::isDone()
{
	return isWin() || isDefeat();
}

float RLMarineScvDefence::getReward() {

	float reward = 0.0;

	if (isWin())
		reward = 1.0;
	else if (isDefeat())
		reward = -1.0;

	return reward;
}


// Etc
bool RLMarineScvDefence::isWin() {
	uList myUnits = INFO.getUnits(myUnitType[0], S);
	uList myUnits_ = INFO.getUnits(myUnitType[1], S);
	myUnits.insert(myUnits.end(), myUnits_.begin(), myUnits_.end());

	int killCount = 0;

	for (auto u : myUnits)
		killCount += u->unit()->getKillCount();

	return killCount == ENEMY_UNIT_COUNT;
}

bool RLMarineScvDefence::isDefeat() {
	uList myUnits = INFO.getUnits(myUnitType[0], S);
	uList enUnits = INFO.getUnits(enUnitType, E);

	return enUnits.size() && !myUnits.size();
}

void RLMarineScvDefence::render()
{
	Position Pos = getAvgPosition(INFO.getUnits(myUnitType[0], S));


	focus(Pos);

	focus(theMap.Center());
}

void RLMarineScvDefence::onUnitDestroy(Unit unit) {
	if (!isDone() && !getIsResetting()) {
		cout << TIME << " 유닛 죽었다. " << unit->getType() << ", " << unit->getKillCount() << ", " << (unit->getPlayer() == S) << endl;

		if (unit->getPlayer() == S)
			myKillCount += unit->getKillCount();
		else
			enemyKillCount += unit->getKillCount();
	}
}