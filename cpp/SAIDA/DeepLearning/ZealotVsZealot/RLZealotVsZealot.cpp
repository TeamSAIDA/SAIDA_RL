/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "RLZealotVsZealot.h"
#include "../../UXManager.h"

using namespace BWML;
using namespace MyBot;

RLZealotVsZealot &RLZealotVsZealot::Instance(string shmName) {
	static RLZealotVsZealot instance(shmName);
	return instance;
}

// Init
void RLZealotVsZealot::init(::google::protobuf::Message *message)
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
		DISTANCE = initReq->move_dist();
		MOVE_ANGLE = initReq->move_angle();

		ACTION_MOVE = 360 / MOVE_ANGLE;
		ACTION_ATTACK = ACTION_MOVE + 3;
		ACTION_HOLD = ACTION_ATTACK + 1;
		ACTION_NOTHING = ACTION_HOLD + 1;
		ACTION_SIZE = ACTION_NOTHING;

		// 시작점은 3시 방향으로 한다. ( DISTANCE )
		Position standardPos = Position(DISTANCE * TILE_SIZE, 0);

		int totalAngle = 0;

		while (totalAngle < 360) {
			MovePosition.push_back(getCirclePosFromPosByDegree(Positions::Origin, standardPos, totalAngle));
			totalAngle += MOVE_ANGLE;
		}

		renderInfo = vector<vector<vector<int>>>(MY_UNIT_COUNT, vector<vector<int>>(MovePosition.size(), vector<int>(5, 0)));
	}
	else {
		ACTION_MOVE = 1;
		ACTION_ATTACK = ACTION_MOVE + 1;
		ACTION_HOLD = ACTION_ATTACK + 1;
		ACTION_NOTHING = ACTION_HOLD + 1;
		ACTION_SIZE = ACTION_NOTHING;
	}

	STEP_FRAME = initReq->frames_per_step();

	for (int i = 0; i < MY_UNIT_COUNT; i++)
		isInvalidAction.push_back(false);

	bw->setVision(E, true);

	lastAction = ACTION_NOTHING;

	myType = Protoss_Zealot;
	enType = Protoss_Zealot;
}

bool RLZealotVsZealot::initializeAndValidate()
{
	if (myUnitSet.empty()) {
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
		for (auto u : INFO.getUnits(myType, S))
			myUnitSet.push_back(u->unit());

		for (auto e : INFO.getUnits(enType, E))
			enUnitSet.push_back(e->unit());
	}

	return !(myUnitSet.empty() && enUnitSet.empty());
}

void RLZealotVsZealot::makeInitMessage(::google::protobuf::Message *initMessage)
{
	InitRes *message = (InitRes *)initMessage;

	message->set_num_action_space(ACTION_SIZE);

	setTypeInfo(message->add_unit_type_map(), Protoss_Zealot);

	for (int i = 0; i < theMap.WalkSize().x; i++)
		for (int j = 0; j < theMap.WalkSize().y; j++)
			message->add_iswalkable(bw->isWalkable(WalkPosition(i, j)));
}


// Reset
void RLZealotVsZealot::reset(bool isFirstResetCall)
{
	for (auto u : myUnitSet)
	{
		if (u->getID() > 9000) {
			restartGame();
			break;
		}
	}

	INFO.clearUnitNBuilding();
	myUnitSet.clear();
	enUnitSet.clear();
}

bool RLZealotVsZealot::isResetFinished()
{
	return INFO.getUnits(myType, S).size() == MY_UNIT_COUNT && INFO.getUnits(enType, E).size() == ENEMY_UNIT_COUNT;
}

// Action
void RLZealotVsZealot::step(::google::protobuf::Message *stepReqMsg)
{
	if (myUnitSet.empty())
		return;

	for (int i = 0; i < MY_UNIT_COUNT; i++) {
		isInvalidAction.at(i) = false;
		Unit u = myUnitSet.at(i);

		if (!u->exists())
		{
			continue;
		}

		Action act = ((StepReq *)stepReqMsg)->action(i);
		UnitInfo *me = INFO.getUnitInfo(u, S);

		if (ACTION_TYPE == 0) {
			me->setLastAction(act.action_num());
			me->setLastCommandPosition(u->getPosition());

			if (act.action_num() < ACTION_MOVE)
			{
				vector<int> myAlts = getNearObstacle(me, ACTION_SIZE);

				Position targetPos = u->getPosition() + (MovePosition[act.action_num()]);

				if (myAlts[act.action_num()] < 50)
					isInvalidAction.at(i) = true;

				if (myAlts[act.action_num()] < DISTANCE)
					targetPos = getDirectionDistancePosition(me->pos(), targetPos, myAlts[act.action_num()]);

				CommandUtil::rightClick(u, targetPos);
			}
			else if (act.action_num() < ACTION_ATTACK)
			{
				int action_idx = act.action_num() - ACTION_MOVE;

				if (enUnitSet.at(action_idx)->exists())
					CommandUtil::attackUnit(u, enUnitSet.at(action_idx));
				else
					isInvalidAction.at(i) = true;
			}
			else if (act.action_num() < ACTION_HOLD)
			{
				CommandUtil::hold(u);
			}
			else//(act.action_num() < ACTION_NOTHING)
			{

			}
		}
		else {
			Position targetPos = Positions::None;

			if (ACTION_TYPE == 1)
				targetPos = { me->pos().x + act.pos_x(), me->pos().y + act.pos_y() };
			else if (ACTION_TYPE == 2)
			{
				int angle = (int)(act.angle() * 360);
				targetPos = getCirclePosFromPosByDegree(me->pos(), me->pos() + Position(0, -act.radius() * TILE_SIZE), angle);
			}

			if (act.action_num() < ACTION_MOVE)
				CommandUtil::move(u, targetPos);
			else if (act.action_num() < ACTION_ATTACK) {
				CommandUtil::attackMove(u, targetPos);
			}
			else if (act.action_num() < ACTION_HOLD)
				CommandUtil::hold(u);
		}
	}
}

bool RLZealotVsZealot::isActionFinished()
{
	if (isDone())
		return true;

	if (INFO.getUnits(myType, S).empty() || INFO.getUnits(enType, E).empty()) {
		bw->setLocalSpeed(200);
		bw->setGUI(true);
		bw->pauseGame();
		cout << "is done 이 아닌데 유닛이 없다?" << endl;

		for (auto u : INFO.getUnits(myType, S)) {
			cout << "my " << u->unit()->getHitPoints() << endl;
		}

		for (auto u : INFO.getUnits(enType, E)) {
			cout << "enemy " << u->unit()->getHitPoints() << endl;
		}

		return false;
	}

	return (TIME - startFrame) % STEP_FRAME == 0;
}

void RLZealotVsZealot::getObservation(::google::protobuf::Message *stateMsg)
{
	State *stateMessage = (State *)stateMsg;

	int idx = 0;

	for (auto u : myUnitSet)
	{
		UInfo *u_info = setUInfo(stateMessage->add_my_unit(), u);

		if (!u->exists())
		{
			for (word i = 0; i < MovePosition.size(); i++)
				TerrainInfo *posInfo = u_info->add_pos_info();

			continue;
		}

		uList myUnits = INFO.getTypeUnitsInRadius(myType, S, u->getPosition(), 7 * TILE_SIZE);
		uList enemies = INFO.getTypeUnitsInRadius(enType, E, u->getPosition(), 7 * TILE_SIZE);

		UnitInfo *me = INFO.getUnitInfo(u, S);
		vector<int> alts = getNearObstacle(me, ACTION_MOVE);
		vector<int> mAngles = getEnemiesInAngle(me, myUnits, ACTION_MOVE, 7 * TILE_SIZE);
		vector<int> eAngles = getEnemiesInAngle(me, enemies, ACTION_MOVE, 7 * TILE_SIZE);

		for (word i = 0; i < MovePosition.size(); i++)
		{
			TerrainInfo *posInfo = u_info->add_pos_info();

			posInfo->set_nearest_obstacle_dist(alts[i]);

			if (alts[i] <= 3 * TILE_SIZE)
				u_info->add_invalid_action(true);
			else
				u_info->add_invalid_action(false);

			// 해당 방향에 적군의 마리수
			posInfo->set_udi_int_1(eAngles[i * 4]);
			// 해당 방향에 가장 가까운 적군의 거리를 넣는다.
			posInfo->set_udi_double_1(eAngles[i * 4] + 1);
			// 해당 방향에 적군의 hp
			posInfo->set_udi_int_3(eAngles[i * 4] + 2);

			// 해당 방향에 아군의 마리수
			posInfo->set_udi_int_2(mAngles[i * 4]);
			// 해당 방향에 가장 가까운 아군의 거리를 넣는다.
			posInfo->set_udi_double_2(mAngles[i * 4 + 1]);

			renderInfo[idx][i][0] = mAngles[i * 4];
			renderInfo[idx][i][1] = mAngles[i * 4 + 1];
			renderInfo[idx][i][2] = eAngles[i * 4];
			renderInfo[idx][i][3] = eAngles[i * 4 + 1];
			renderInfo[idx][i][4] = alts[i];
		}

		if (u->getGroundWeaponCooldown() > 0 || INFO.getTypeUnitsInRadius(enType, E, u->getPosition(), 5 * TILE_SIZE).size() == 0)
			u_info->add_invalid_action(true);
		else
			u_info->add_invalid_action(false);

		u_info->add_invalid_action(false); // Hold
		u_info->add_invalid_action(false); // Nothing

		idx++;
	}

	for (auto u : enUnitSet)
		UInfo *u_info = setUInfo(stateMessage->add_en_unit(), u);
}

// Done, Reward
bool RLZealotVsZealot::isDone()
{
	return isWin() || isDefeat();
}


float RLZealotVsZealot::getReward() {

	float reward = 0.0;

	if (isWin())
		return WIN_REWARD;
	else if (isDefeat())
		return LOSE_REWARD;

	return reward;
}


// Etc
bool RLZealotVsZealot::isWin() {
	for (auto e : enUnitSet)
		if (e->exists())
			return false;

	return true;
}

bool RLZealotVsZealot::isDefeat() {
	for (auto s : myUnitSet)
		if (s->exists())
			return false;

	return true;
}

void RLZealotVsZealot::render()
{
	vector<UnitInfo *> drawUnit;

	for (auto u : myUnitSet) {
		if (!u->exists())
			continue;

		UnitInfo *v = INFO.getUnitInfo(u, S);

		if (u->getOrderTargetPosition().isValid())
			bw->drawLineMap(u->getPosition(), u->getOrderTargetPosition(), u->getLastCommand().getType() == UnitCommandTypes::Attack_Unit
							|| u->getLastCommand().getType() == UnitCommandTypes::Attack_Move ? Colors::Red : Colors::Yellow);

		UXManager::Instance().drawUnitHP(v);

		drawUnit.push_back(v);
	}

	// focusing
	int idx = (stepNum / 240) % (drawUnit.size() + 1);

	if (idx == drawUnit.size()) {
		Position focusPos = Positions::Origin;
		idx = 0;

		for (auto v : drawUnit) {
			focusPos += v->pos();

			if (ACTION_TYPE == 0) {
				for (word i = 0; i < MovePosition.size(); i++)
				{
					Position drawP = getDirectionDistancePosition(v->getLastCommandPosition(), v->getLastCommandPosition() + MovePosition[i], renderInfo[idx][i][4]);
					bw->drawLineMap(v->getLastCommandPosition(), drawP, Colors::White);
					Position p1, p2, p3, p4, p5;
					p1 = v->getLastCommandPosition() + (MovePosition[i] * 1 / DISTANCE);
					p2 = v->getLastCommandPosition() + (MovePosition[i] * 2 / DISTANCE);
					p3 = v->getLastCommandPosition() + (MovePosition[i] * 3 / DISTANCE);
					p4 = v->getLastCommandPosition() + (MovePosition[i] * 4 / DISTANCE);
					p5 = v->getLastCommandPosition() + (MovePosition[i] * 5 / DISTANCE);

					if (renderInfo[idx][i][0])
						bw->drawTextMap(p1, "%d", renderInfo[idx][i][0]);

					if (renderInfo[idx][i][1])
						bw->drawTextMap(p2, "%d", renderInfo[idx][i][1]);

					if (renderInfo[idx][i][2])
						bw->drawTextMap(p3, "%d", renderInfo[idx][i][2]);

					if (renderInfo[idx][i][3])
						bw->drawTextMap(p4, "%d", renderInfo[idx][i][3]);

					if (renderInfo[idx][i][4])
						bw->drawTextMap(p5, "%d", renderInfo[idx][i][4]);
				}
			}

			if (v->getLastAction() == ACTION_SIZE)
			{
				bw->drawCircleMap(v->pos(), 10, Colors::Blue, true);
			}
			else
			{
				bw->drawCircleMap(v->getLastCommandPosition(), 5, Colors::Cyan, true);
				bw->drawCircleMap(lastTargetPos, 5, Colors::Blue, true);
			}

			idx++;
		}

		if (drawUnit.size())
			focus(focusPos / drawUnit.size());
	}
	else {
		UnitInfo *v = drawUnit.at(idx);

		if (ACTION_TYPE == 0) {
			for (word i = 0; i < MovePosition.size(); i++)
			{
				Position drawP = getDirectionDistancePosition(v->getLastCommandPosition(), v->getLastCommandPosition() + MovePosition[i], renderInfo[idx][i][4]);
				bw->drawLineMap(v->getLastCommandPosition(), drawP, Colors::White);
				Position p1, p2, p3, p4, p5;
				p1 = v->getLastCommandPosition() + (MovePosition[i] * 1 / DISTANCE);
				p2 = v->getLastCommandPosition() + (MovePosition[i] * 2 / DISTANCE);
				p3 = v->getLastCommandPosition() + (MovePosition[i] * 3 / DISTANCE);
				p4 = v->getLastCommandPosition() + (MovePosition[i] * 4 / DISTANCE);
				p5 = v->getLastCommandPosition() + (MovePosition[i] * 5 / DISTANCE);

				if (renderInfo[idx][i][0])
					bw->drawTextMap(p1, "%d", renderInfo[idx][i][0]);

				if (renderInfo[idx][i][1])
					bw->drawTextMap(p2, "%d", renderInfo[idx][i][1]);

				if (renderInfo[idx][i][2])
					bw->drawTextMap(p3, "%d", renderInfo[idx][i][2]);

				if (renderInfo[idx][i][3])
					bw->drawTextMap(p4, "%d", renderInfo[idx][i][3]);

				if (renderInfo[idx][i][4])
					bw->drawTextMap(p5, "%d", renderInfo[idx][i][4]);
			}
		}

		if (v->getLastAction() == ACTION_SIZE)
		{
			bw->drawCircleMap(v->pos(), 10, Colors::Blue, true);
		}
		else
		{
			bw->drawCircleMap(v->getLastCommandPosition(), 5, Colors::Cyan, true);
			bw->drawCircleMap(lastTargetPos, 5, Colors::Blue, true);
		}

		focus(v->pos());
	}

	for (auto e : enUnitSet) {
		if (!e->exists())
			continue;

		UXManager::Instance().drawUnitHP(INFO.getUnitInfo(e, E));
	}
}

void RLZealotVsZealot::onUnitDestroy(Unit unit) {
}
