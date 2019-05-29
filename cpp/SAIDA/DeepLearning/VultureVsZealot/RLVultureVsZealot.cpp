/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "RLVultureVsZealot.h"
#include "../../UXManager.h"

using namespace BWML;
using namespace MyBot;

RLVultureVsZealot &RLVultureVsZealot::Instance(string shmName) {
	static RLVultureVsZealot instance(shmName);
	return instance;
}

// Init
void RLVultureVsZealot::init(::google::protobuf::Message *message)
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

	MY_UNIT_COUNT = 1;

	for (int i = 0; i < MY_UNIT_COUNT; i++)
		isInvalidAction.push_back(false);

	if (map_version == 0 || map_version == 2)
		ENEMY_UNIT_COUNT = 1;
	else if (map_version == 1 || map_version == 3)
		ENEMY_UNIT_COUNT = 2;

	bw->setVision(E, true);

	lastAction = ACTION_SIZE;
}

bool RLVultureVsZealot::initializeAndValidate()
{
	if (agent == nullptr) {
		INFO.clearUnitNBuilding();

		for (auto unit : bw->getAllUnits()) {
			if (unit->exists()) {
				INFO.onUnitCreate(unit);
				INFO.onUnitComplete(unit);
			}
		}

		INFO.update();
	}

	if (getIsResetting() && isResetFinished()) {
		for (auto v : INFO.getUnits(Terran_Vulture, S)) {
			idOrderedList.push_back(v->id());
			agent = v->unit();
		}

		for (auto z : INFO.getUnits(Protoss_Zealot, E)) {
			enemyIdOrderedList.push_back(z->id());
		}
	}

	return !(INFO.getUnits(Terran_Vulture, S).empty() && INFO.getUnits(Protoss_Zealot, E).empty());
}

void RLVultureVsZealot::makeInitMessage(::google::protobuf::Message *initMessage)
{
	InitRes *message = (InitRes *)initMessage;

	if (ACTION_TYPE == 0)
		message->set_num_action_space(ACTION_SIZE + 1);
	else
		message->set_num_action_space(2);

	setTypeInfo(message->add_unit_type_map(), Terran_Vulture);
	setTypeInfo(message->add_unit_type_map(), Protoss_Zealot);

	for (int i = 0; i < theMap.WalkSize().x; i++)
		for (int j = 0; j < theMap.WalkSize().y; j++)
			message->add_iswalkable(bw->isWalkable(WalkPosition(i, j)));
}


// Reset
void RLVultureVsZealot::reset(bool isFirstResetCall)
{
	if (!idOrderedList.empty() && idOrderedList.at(0) > 9000)
		restartGame();

	INFO.clearUnitNBuilding();

	agent = nullptr;
	idOrderedList.clear();
	enemyIdOrderedList.clear();

	myKillCount = 0;
	enemyKillCount = 0;
}

bool RLVultureVsZealot::isResetFinished()
{
	return INFO.getUnits(Terran_Vulture, S).size() == MY_UNIT_COUNT && INFO.getUnits(Protoss_Zealot, E).size() == ENEMY_UNIT_COUNT;
}


// Action
void RLVultureVsZealot::step(::google::protobuf::Message *stepReqMsg)
{
	if (HUMAN_MODE)
	{
		return;
	}

	UnitInfo *s = INFO.getUnitInfo(agent, S);

	if (!s)
		return;

	Position targetPos = Positions::None;
	int i = 0;

	for (Action act : ((StepReq *)stepReqMsg)->action()) // 어차피 하나
	{
		isInvalidAction.at(i) = false;


		if (ACTION_TYPE == 0)
		{
			if (act.action_num() == ACTION_SIZE)
			{
				Unit target = INFO.getClosestTypeUnit(E, s->pos(), Protoss_Zealot)->unit();

				isInvalidAction.at(i) = s->unit()->getGroundWeaponCooldown() > 0 || !agent->isInWeaponRange(target);

				if (agent->getDistance(target) < 2 * TILE_SIZE)
					CommandUtil::patrol(agent, getDirectionDistancePosition(target->getPosition(), agent->getPosition(), -2 * TILE_SIZE));
				else {
					double headingRadian = (getRadian(agent->getPosition(), target->getPosition()) - agent->getAngle()) / 2;

					if (headingRadian > M_PI_2 || headingRadian < -M_PI_2)
						headingRadian += M_PI;

					headingRadian = agent->getAngle() + headingRadian;

					while (headingRadian < 0)
						headingRadian += 2 * M_PI;

					while (headingRadian > 2 * M_PI)
						headingRadian -= 2 * M_PI;

					// 사잇각의 좌표로 p컨
					CommandUtil::patrol(agent, getPosByPosDistRadian(agent->getPosition(), 2 * TILE_SIZE, headingRadian));
				}
			}
			else
			{
				vector<int> myAlts = getNearObstacle(s, ACTION_SIZE);

				targetPos = s->pos() + MovePosition[act.action_num()];

				if (myAlts[act.action_num()] < 50)
					isInvalidAction.at(i) = true;

				if (myAlts[act.action_num()] < DISTANCE)
					targetPos = getDirectionDistancePosition(s->pos(), targetPos, myAlts[act.action_num()]);

				CommandUtil::rightClick(s->unit(), targetPos);
				commandPosition = s->pos();
			}
		}
		else {
			if (ACTION_TYPE == 1)
				targetPos = { s->pos().x + act.pos_x(), s->pos().y + act.pos_y() };
			else if (ACTION_TYPE == 2)
			{
				int angle = (int)(act.angle() * 360);
				targetPos = getCirclePosFromPosByDegree(s->pos(), s->pos() + Position(0, -act.radius() * TILE_SIZE), angle);
			}

			if (act.action_num() == 0)
				CommandUtil::move(s->unit(), targetPos);
			else if (act.action_num() == 1) {
				isInvalidAction.at(i) = s->unit()->getGroundWeaponCooldown() > 0;
				CommandUtil::patrol(s->unit(), targetPos);
			}
		}

		lastAction = act.action_num();
		preCooldown = s->unit()->getGroundWeaponCooldown();
		lastTargetPos = targetPos;
	}

	// 9 : do nothing...
	return;
}

bool RLVultureVsZealot::isActionFinished()
{
	if (HUMAN_MODE)
		return true;

	if (isDone())
		return true;

	if (INFO.getUnits(Terran_Vulture, S).empty() || INFO.getUnits(Protoss_Zealot, E).empty())
		return false;

	// Attack Command 유효성 체크
	if (ACTION_TYPE == 0 && lastAction == ACTION_SIZE || ACTION_TYPE != 0 && lastAction == 1)
	{
		if (preCooldown < agent->getGroundWeaponCooldown()) {
			isInvalidAction.at(0) = false;
		}

		preCooldown = agent->getGroundWeaponCooldown();
	}

	if (STEP_FRAME != -1)
		return (TIME - startFrame) % STEP_FRAME == 0;

	// Attack Command 였다면
	if (ACTION_TYPE == 0 && lastAction == ACTION_SIZE || ACTION_TYPE != 0 && lastAction == 1)
	{
		if (!isInvalidAction.at(0)) {
			preCooldown = -1;
			return true;
		}
	}
	else
	{
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
	}

	return false;
}

void RLVultureVsZealot::getObservation(::google::protobuf::Message *stateMsg)
{
	State *stateMessage = (State *)stateMsg;

	uList vultures = INFO.getUnits(Terran_Vulture, S);

	for (auto v : vultures)
	{
		UInfo *v_info = setUInfo(stateMessage->add_my_unit(), v);

		if (ACTION_TYPE == 0)
			for (auto alt : getNearObstacle(v, ACTION_SIZE))
			{
				TerrainInfo *posInfo = v_info->add_pos_info();

				posInfo->set_nearest_obstacle_dist(alt);
			}
	}

	uList zealots = INFO.getUnits(Protoss_Zealot, E);

	for (auto z : zealots)
		UInfo *z_info = setUInfo(stateMessage->add_en_unit(), z);
}


// Done, Reward
bool RLVultureVsZealot::isDone()
{
	return isWin() || isDefeat();
}

float RLVultureVsZealot::getReward() {

	float reward = 0.0;

	if (isWin())
		reward = 1.0;
	else if (isDefeat())
		reward = -1.0;

	return reward;
}


// Etc
bool RLVultureVsZealot::isWin() {
	uList vultures = INFO.getUnits(Terran_Vulture, S);

	int killCount = 0;

	for (auto v : vultures)
		killCount += v->unit()->getKillCount();

	return myKillCount + killCount == ENEMY_UNIT_COUNT;
}

bool RLVultureVsZealot::isDefeat() {
	uList zealots = INFO.getUnits(Protoss_Zealot, E);

	int killCount = 0;

	for (auto z : zealots)
		killCount += z->unit()->getKillCount();

	return enemyKillCount + killCount == MY_UNIT_COUNT;
}

void RLVultureVsZealot::render()
{
	UnitInfo *self = INFO.getUnitInfo(agent, S);

	if (self) {
		focus(self->pos());

		vector<int> myAlts = getNearObstacle(self, ACTION_SIZE);

		for (int i = 0; i < ACTION_SIZE; i++) {
			int a = myAlts[i];
			Position end = self->pos() + MovePosition[i];
			bw->drawLineMap(self->pos(), end, Colors::White);
			bw->drawTextMap(end, "%d", a);
		}

		if (agent->getOrderTargetPosition().isValid())
			bw->drawLineMap(agent->getPosition(), agent->getOrderTargetPosition(), agent->getLastCommand().getType() == UnitCommandTypes::Patrol ? Colors::Red : Colors::Yellow);
	}
	else
		focus(theMap.Center());

	uList vultures = INFO.getUnits(Terran_Vulture, S);
	uList zealots = INFO.getUnits(Protoss_Zealot, E);

	for (auto v : vultures) {
		UXManager::Instance().drawUnitHP(v);
	}

	for (auto z : zealots)
		UXManager::Instance().drawUnitHP(z);
}

void RLVultureVsZealot::onUnitDestroy(Unit unit) {
	if (!isDone() && !getIsResetting()) {
		if (unit->getPlayer() == S)
			myKillCount += unit->getKillCount();
		else
			enemyKillCount += unit->getKillCount();
	}
}