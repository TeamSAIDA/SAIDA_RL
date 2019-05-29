/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "SaidaUtil.h"
#include <stdio.h>

using namespace MyBot;

Position MyBot::getAvgPosition(uList units)
{
	Position avgPos = Positions::Origin;

	for (auto u : units)
		avgPos += u->pos();

	return (Position)(avgPos / units.size());
}

bool MyBot::isUseMapSettings()
{
	return Broodwar->getGameType() == GameTypes::Use_Map_Settings ? true : false;
}

void MyBot::focus(Position pos) {
	if (Config::Debug::Focus)
		bw->setScreenPosition(pos - Position(320, 160));
}

void MyBot::restartGame() {
	Config::BWAPIOptions::RestartGame = true;
	bw->restartGame();
}

void MyBot::leaveGame() {
	Config::BWAPIOptions::EndGame = true;
	bw->leaveGame();
}

bool MyBot::isSameArea(UnitInfo *u1, UnitInfo *u2) {
	return u1->pos().isValid() && u2->pos().isValid() && isSameArea(theMap.GetArea((WalkPosition)u1->pos()), theMap.GetArea((WalkPosition)u2->pos()));
}

bool MyBot::isSameArea(Position a1, Position a2) {
	return a1.isValid() && a2.isValid() && isSameArea(theMap.GetArea((WalkPosition)a1), theMap.GetArea((WalkPosition)a2));
}

bool MyBot::isSameArea(TilePosition a1, TilePosition a2) {
	return a1.isValid() && a2.isValid() && isSameArea(theMap.GetArea(a1), theMap.GetArea(a2));
}

bool MyBot::isSameArea(const Area *a1, const Area *a2) {
	if (a1 == nullptr || a2 == nullptr)
		return false;

	if (a1->Id() == a2->Id())
		return true;

	return false;
}

bool MyBot::isBlocked(Unit unit, int size) {
	// 시야에 없는 유닛에 대해서는 체크하지 않는다.
	if (!unit->exists())
		return false;

	return isBlocked(unit->getTop(), unit->getLeft(), unit->getBottom(), unit->getRight(), size);
}

bool MyBot::isBlocked(const UnitType unitType, Position centerPosition, int size) {
	return isBlocked(centerPosition.y - unitType.dimensionUp(), centerPosition.x - unitType.dimensionLeft(), centerPosition.y + unitType.dimensionDown() + 1, centerPosition.x + unitType.dimensionRight() + 1, size);
}

bool MyBot::isBlocked(const UnitType unitType, TilePosition topLeft, int size) {
	TilePosition bottomRight = topLeft + unitType.tileSize();
	return isBlocked(topLeft.y * 32, topLeft.x * 32, bottomRight.y * 32, bottomRight.x * 32, size);
}

bool MyBot::isBlocked(int top, int left, int bottom, int right, int size) {
	Position center = Position((left + right) / 2, (top + bottom) / 2);

	if (getAltitude(center + Position(0, size)) > size || getAltitude(center + Position(0, -size)) > size
			|| getAltitude(center + Position(size, 0)) > size || getAltitude(center + Position(-size, 0)) > size)
		return false;

	int minX = left / 8;
	int minY = top / 8;
	int maxY = bottom / 8;
	int maxX = right / 8;
	int x = left / 8;
	int y = top / 8;

	altitude_t beforeAltitude = getAltitude(WalkPosition(x, y));
	altitude_t firstAltitude = beforeAltitude;

	int blockedCnt = firstAltitude < size ? 1 : 0;
	int smallCnt = 0;

	for (x++; x < maxX; x++) {
		altitude_t altitude = getAltitude(WalkPosition(x, y));

		if (beforeAltitude >= size && altitude < size)
			blockedCnt++;

		if (size > altitude)
			smallCnt++;

		beforeAltitude = altitude;
	}

	for (x--; y < maxY; y++) {
		altitude_t altitude = getAltitude(WalkPosition(x, y));

		if (beforeAltitude >= size && altitude < size)
			blockedCnt++;

		if (size > altitude)
			smallCnt++;

		beforeAltitude = altitude;
	}

	for (y--; x >= minX; x--) {
		altitude_t altitude = getAltitude(WalkPosition(x, y));

		if (beforeAltitude >= size && altitude < size)
			blockedCnt++;

		if (size > altitude)
			smallCnt++;

		beforeAltitude = altitude;
	}

	for (x++; y > minY; y--) {
		altitude_t altitude = getAltitude(WalkPosition(x, y));

		if (beforeAltitude >= size && altitude < size)
			blockedCnt++;

		if (size > altitude)
			smallCnt++;

		beforeAltitude = altitude;
	}

	if (firstAltitude < size && beforeAltitude < size && blockedCnt > 1)
		blockedCnt--;

	bool narrow = false;

	if (blockedCnt == 1) {
		narrow = smallCnt > 2 * (maxX - minX + maxY - minY - 1) * 0.7;
		//cout << 2 * (maxX - minX + maxY - minY - 1) * 0.7 << endl;
	}

	return narrow || blockedCnt > 1;
}

bool MyBot::isValidPath(Position s, Position e)
{
	TilePosition TS = TilePosition(s);
	TilePosition TE = TilePosition(e);
	WalkPosition WS = WalkPosition(s);
	WalkPosition WE = WalkPosition(e);

	// not walkable
	if (!s.isValid() || !e.isValid() || getAltitude(WE) <= 0) {
		return false;
	}

	// 건물이나 미네랄 있으면 False;
	Unitset tmp = Broodwar->getUnitsInRadius(e, 40, Filter::IsBuilding || Filter::IsNeutral);
	// 상대 지상 유닛이 있으면 False;
	uList eUnits = INFO.getUnitsInRadius(E, e, 30, true, false, false);

	if (tmp.size() || eUnits.size())
		return false;

	return true;
}

int MyBot::getPathValue(Position st, Position en)
{
	TilePosition TS = TilePosition(st);
	TilePosition TE = TilePosition(en);
	WalkPosition WS = WalkPosition(st);
	WalkPosition WE = WalkPosition(en);

	int point = 0;

	// Not walkable 또는 장애물이 있는 경우
	if (theMap.Valid(en) == false || getAltitude(WE) <= 0 ||
			Broodwar->getUnitsInRadius(en, 16, Filter::IsBuilding || Filter::IsNeutral).size())
	{
		return -1;
	}

	bool nearChoke = false;

	if (theMap.GetArea(TS) == nullptr || theMap.GetArea(TE) == nullptr)
	{
		nearChoke = true;
	}
	else
	{
		if (theMap.GetArea(TS) != theMap.GetArea(TE) && Broodwar->getGroundHeight(TS) != Broodwar->getGroundHeight(TE))
		{
			int dist = 0;
			theMap.GetPath(st, en, &dist);

			if (dist == -1 || dist > st.getApproxDistance(en) * 2)
				return -1;
			else // 영역 다른데 높이도 다른데 거리는 가까워... ChokePoint 근처다.
				nearChoke = true;
		}
	}

	int dangerPoint = 0;
	getDangerUnitNPoint(en, &dangerPoint, false);
	point = 2 * dangerPoint;

	if (nearChoke == false)
	{
		point = point + getAltitude(WE);
	}
	else {
		int chokeAlt = getAltitude(WE) < 100 ? 100 : getAltitude(WE);
		point = point + chokeAlt;
	}

	return point;
}

int MyBot::getGroundDistance(Position st, Position en)
{
	int dist = 0;
	theMap.GetPath(st, en, &dist);
	return dist;
}

// invalid 할 경우 -1 return;
int MyBot::getAltitude(Position pos)
{
	return getAltitude((WalkPosition)pos);
}

int MyBot::getAltitude(TilePosition pos)
{
	return getAltitude((WalkPosition)pos);
}

int MyBot::getAltitude(WalkPosition pos)
{
	return pos.isValid() ? theMap.GetMiniTile(pos).Altitude() : -1;
}

int MyBot::getPathValueForAir(Position en)
{	// Not walkable 또는 장애물이 있는 경우
	if (theMap.Valid(en) == false)
		return -1;

	int dangerPoint = 0;
	getDangerUnitNPoint(en, &dangerPoint, true);

	return dangerPoint;
}

// 해당 위치 Pos에서 아군 지상 유닛이 위험한 정도를 점수로 환산
// Range Unit의 사거리를 계산하여 공격 범위로부터 멀수록 높은 포인트를 줘보자.
// 가장 위협적인 Unit에 일꾼을 돌려주진 않는다. 단 아무공격 유닛이 없을때는 일꾼의 위험도를 돌려준다.
// 이는 일꾼 카이팅을 위함.
UnitInfo *MyBot::getDangerUnitNPoint(Position pos, int *point, bool isFlyer)
{
	uList enemyUnits = INFO.getUnitsInRadius(E, pos, 18 * TILE_SIZE, true, true, false, true);
	uList enemyDefence = INFO.getDefenceBuildingsInRadius(E, pos, 16 * TILE_SIZE, false, true);

	if (isFlyer)
	{
		if (INFO.enemyRace == Races::Terran)
		{
			for (auto b : INFO.getTypeBuildingsInRadius(Terran_Missile_Turret, E, pos, 16 * TILE_SIZE, false, true))
				enemyDefence.push_back(b);
		}

		if (INFO.enemyRace == Races::Zerg)
		{
			for (auto b : INFO.getTypeBuildingsInRadius(Zerg_Spore_Colony, E, pos, 16 * TILE_SIZE, false, true))
				enemyDefence.push_back(b);
		}
	}

	int min_gap = NOT_DANGER;
	UnitInfo *dangerUnit = nullptr;

	for (auto eu : enemyUnits)
	{
		int weaponRange = isFlyer ? E->weaponMaxRange(eu->type().airWeapon()) : E->weaponMaxRange(eu->type().groundWeapon());

		if (!isFlyer && eu->type() == Terran_Siege_Tank_Tank_Mode && (INFO.hasResearched(TechTypes::Tank_Siege_Mode) || S->hasResearched(TechTypes::Tank_Siege_Mode)))
			weaponRange = WeaponTypes::Arclite_Shock_Cannon.maxRange(); // 시즈업 됐으면 기본적으로 탱크모드 사정거리를 적용한다.

		if (eu->type() == Protoss_Carrier || eu->type() == Zerg_Scourge)
			weaponRange = 8 * TILE_SIZE;

		if (weaponRange == 0 || eu->type() == Protoss_Arbiter)
			continue;

		/// 임시로 일단 조치하고 해보자
		// Wraith는 지상유닛에서는 빼본다.... 별거 아닌 놈이라서
		if (!isFlyer && (eu->type() == Terran_Wraith || eu->type() == Protoss_Scout || eu->type() == Protoss_Carrier || eu->type() == Protoss_Interceptor))
			continue;

		if (eu->type() == Zerg_Lurker)
		{
			if (!eu->isBurrowed())
				continue;
		}
		else
		{
			if (eu->isBurrowed())
				continue;
		}

		int gap = pos.getApproxDistance(eu->pos()) - weaponRange;

		if (min_gap > gap)
		{
			min_gap = gap;
			dangerUnit = eu;
		}
	}

	for (auto eb : enemyDefence)
	{
		int weaponRange = isFlyer ? eb->type().airWeapon().maxRange() : eb->type().groundWeapon().maxRange();

		int gap = pos.getApproxDistance(eb->pos()) - weaponRange;

		if (min_gap > gap)
		{
			min_gap = gap;
			dangerUnit = eb;
		}
	}

	//	if (dangerUnit == nullptr)
	if (!isFlyer)
	{
		uList enemyWorkers = INFO.getTypeUnitsInRadius(INFO.getWorkerType(INFO.enemyRace), E, pos, 10 * TILE_SIZE);

		for (auto ew : enemyWorkers)
		{
			int gap = pos.getApproxDistance(ew->pos()) - ew->type().groundWeapon().maxRange();

			if (min_gap > gap)
			{
				min_gap = gap;
			}
		}
	}

	*point = min_gap;

	return dangerUnit;
}

vector<Position> MyBot::getWidePositions(Position source, Position target, bool forward, int gap, int angle, int cnt)
{
	vector<Position> l;

	Position defencePos = source;

	Position forward_pos = { 0, 0 };
	// Back Move 하는 거리를 backDistaceNeed 로 통일 시키기 위한 Alpha값 찾기
	double forwardDistaceNeed = gap;
	double distance = defencePos.getDistance(target);
	double alpha = forwardDistaceNeed / distance;

	if (forward)
	{
		forward_pos.x = defencePos.x - (int)((defencePos.x - target.x) * alpha);
		forward_pos.y = defencePos.y - (int)((defencePos.y - target.y) * alpha);
	}
	else // Backend
	{
		forward_pos.x = defencePos.x + (int)((defencePos.x - target.x) * alpha);
		forward_pos.y = defencePos.y + (int)((defencePos.y - target.y) * alpha);
	}

	l.push_back(forward_pos);
	//	Broodwar->drawCircleMap(forward_pos, 2, Colors::Cyan, true);
	vector<double> degrees;

	for (int i = 0; i < cnt - 1; i++)
	{
		if (i % 2 == 0)
			degrees.push_back(angle * ((i / 2) + 1));
		else
			degrees.push_back(-angle * ((i / 2) + 1));
	}

	for (int pos_idx = 0; pos_idx < cnt - 1; pos_idx++)
	{
		Position tmp_pos = getCirclePosFromPosByDegree(defencePos, forward_pos, degrees[pos_idx]);
		//		Broodwar->drawCircleMap(tmp_pos, 2, Colors::Cyan, true);
		l.push_back(tmp_pos);
	}

	return l;
}

vector<Position> MyBot::getRoundPositions(Position source, int gap, int angle)
{
	vector<Position> l;
	Position pos = source + Position(gap, 0);

	int total_angle = 0;

	for (int i = 0; i < 360; i++)
	{
		if (total_angle >= 360)
			break;

		pos = getCirclePosFromPosByDegree(source, pos, angle);
		total_angle += angle;

		if (theMap.Valid(pos) == false || getAltitude(pos) <= 0 ||
				bw->getUnitsInRadius(pos, TILE_SIZE, (Filter::IsMineralField || Filter::IsRefinery || (Filter::IsBuilding && !Filter::IsFlyingBuilding))).size())
			continue;

		l.push_back(pos);
	}

	return l;
}

Position MyBot::getDirectionDistancePosition(Position source, Position direction, int distance)
{
	Position forward_pos = { 0, 0 };
	// Back Move 하는 거리를 backDistaceNeed 로 통일 시키기 위한 Alpha값 찾기
	double forwardDistanceNeed = distance;
	double dist = source.getDistance(direction);
	double alpha = forwardDistanceNeed / dist;

	forward_pos.x = source.x - (int)((source.x - direction.x) * alpha);
	forward_pos.y = source.y - (int)((source.y - direction.y) * alpha);

	return forward_pos;
}

Position MyBot::getBackPostion(UnitInfo *uInfo, Position ePos, int length, bool avoidUnit)
{
	Position back_pos = { 0, 0 };
	Position myPos = uInfo->pos();

	// Back Move 하는 거리를 backDistaceNeed 로 통일 시키기 위한 Alpha값 찾기
	double distance = myPos.getDistance(ePos);
	double alpha = (double)length / distance;

	back_pos.x = myPos.x + (int)((myPos.x - ePos.x) * alpha);
	back_pos.y = myPos.y + (int)((myPos.y - ePos.y) * alpha);

	Position standardPos = back_pos;

	int total_angle = 0;

	int value = uInfo->type().isFlyer() ? getPathValueForAir(myPos) : getPathValue(myPos, myPos);
	Position bestPos = myPos;

	while (total_angle < 360)
	{
		back_pos = getCirclePosFromPosByDegree(myPos, standardPos, total_angle);
		total_angle += 30;

		if (avoidUnit)
		{
			Position behind = getDirectionDistancePosition(myPos, back_pos, 2 * TILE_SIZE);

			if (INFO.getUnitsInRadius(S, behind, (int)(1.5 * TILE_SIZE), true, false, false).size())
				continue;
		}

		int tmp_val = uInfo->type().isFlyer() ? getPathValueForAir(back_pos) : getPathValue(myPos, back_pos);

		if (value < tmp_val)
		{
			value = tmp_val;
			bestPos = back_pos;
		}
	}

	return bestPos;
}

// Unit, My Unit Position, Enemy Unit Position, move Distance
void MyBot::moveBackPostion(UnitInfo *uInfo, Position ePos, int length)
{
	if (uInfo->frame() >= TIME)
		return;

	Position back_pos = { 0, 0 };
	Position myPos = uInfo->pos();

	// Back Move 하는 거리를 backDistaceNeed 로 통일 시키기 위한 Alpha값 찾기
	double distance = myPos.getDistance(ePos);
	double alpha = (double)length / distance;

	back_pos.x = myPos.x + (int)((myPos.x - ePos.x) * alpha);
	back_pos.y = myPos.y + (int)((myPos.y - ePos.y) * alpha);

	int angle = 30;
	int total_angle = 0;

	int value = uInfo->type().isFlyer() ? getPathValueForAir(back_pos) : getPathValue(myPos, back_pos);
	Position bPos = back_pos;

	while (total_angle <= 360)
	{
		back_pos = getCirclePosFromPosByDegree(myPos, back_pos, angle);
		total_angle += angle;

		int tmp_val = uInfo->type().isFlyer() ? getPathValueForAir(back_pos) : getPathValue(myPos, back_pos);

		if (value < tmp_val)
		{
			value = tmp_val;
			bPos = back_pos;
		}
	}

	uInfo->unit()->move(bPos);
}

Position MyBot::findRandomeSpot(Position p) {
	int cnt = 0;

	while (cnt < 10) {
		Position rPos(2 * rand() % 65 - 32, 2 * rand() % 65 - 32);
		Position newP = p + rPos;

		if (getAltitude(newP) > 0)
			return newP;

		cnt++;
	}

	return p;
}

Position MyBot::getCirclePosFromPosByDegree(Position center, Position fromPos, double degree)
{
	return getCirclePosFromPosByRadian(center, fromPos, (degree * M_PI / 180));
}

Position MyBot::getCirclePosFromPosByRadian(Position center, Position fromPos, double radian)
{
	int x = (int)((double)(fromPos.x - center.x) * cos(radian) - (double)(fromPos.y - center.y) * sin(radian) + center.x);
	int y = (int)((double)(fromPos.x - center.x) * sin(radian) + (double)(fromPos.y - center.y) * cos(radian) + center.y);

	return Position(x, y);
}

Position MyBot::getPosByPosDistDegree(Position pos, int dist, double degree)
{
	return getPosByPosDistRadian(pos, dist, (degree * M_PI / 180));
}

Position MyBot::getPosByPosDistRadian(Position pos, int dist, double radian)
{
	int x = pos.x + (int)(dist * cos(radian));
	int y = pos.y + (int)(dist * sin(radian));
	return Position(x, y);
}

double MyBot::getRadian(Position p1, Position p2) {
	Position p = p2 - p1;

	double radian = atan2(-p.y, p.x);

	if (radian < 0)
		radian += 2 * M_PI;

	return radian;
}

double MyBot::getRadian2(Position p1, Position p2) {
	Position p = Position(p2.x - p1.x, p1.y - p2.y);
	double radian = atan2(p.x, p.y);

	if (radian < 0)
		radian += 2 * M_PI;

	return radian;
}

int MyBot::getDamage(Unit attacker, Unit target)
{
	return Broodwar->getDamageFrom(attacker->getType(), target->getType(), attacker->getPlayer(), target->getPlayer());
}

int MyBot::getDamage(UnitType attackerType, UnitType targetType, Player attackerPlayer, Player targetPlayer)
{
	return Broodwar->getDamageFrom(attackerType, targetType, attackerPlayer, targetPlayer);
}

int MyBot::getAttackDistance(int aLeft, int aTop, int aRight, int aBottom, int tLeft, int tTop, int tRight, int tBottom) {
	// compute x distance
	int xDist = aLeft - tRight;

	if (xDist < 0)
	{
		xDist = tLeft - aRight;

		if (xDist < 0)
			xDist = 0;
	}

	// compute y distance
	int yDist = aTop - tBottom;

	if (yDist < 0)
	{
		yDist = tTop - aBottom;

		if (yDist < 0)
			yDist = 0;
	}

	// compute actual distance
	return Positions::Origin.getApproxDistance(Position(xDist, yDist));
}

int MyBot::getAttackDistance(Unit attacker, Unit target) {
	return getAttackDistance(attacker->getLeft(), attacker->getTop(), attacker->getRight(), attacker->getBottom(), target->getLeft() - 1, target->getTop() - 1, target->getRight() + 1, target->getBottom() + 1);
}

int MyBot::getAttackDistance(Unit attacker, UnitType targetType, Position targetPosition) {
	return getAttackDistance(attacker->getLeft(), attacker->getTop(), attacker->getRight(), attacker->getBottom(), targetPosition.x - targetType.dimensionLeft() - 1, targetPosition.y - targetType.dimensionUp() - 1, targetPosition.x + targetType.dimensionRight() + 1, targetPosition.y + targetType.dimensionDown() + 1);
}
int MyBot::getAttackDistance(UnitType attackerType, Position attackerPosition, Unit target) {
	return getAttackDistance(attackerPosition.x - attackerType.dimensionLeft(), attackerPosition.y - attackerType.dimensionUp(), attackerPosition.x + attackerType.dimensionRight(), attackerPosition.y + attackerType.dimensionDown(), target->getLeft() - 1, target->getTop() - 1, target->getRight() + 1, target->getBottom() + 1);
}
int MyBot::getAttackDistance(UnitType attackerType, Position attackerPosition, UnitType targetType, Position targetPosition) {
	return getAttackDistance(attackerPosition.x - attackerType.dimensionLeft(), attackerPosition.y - attackerType.dimensionUp(), attackerPosition.x + attackerType.dimensionRight(), attackerPosition.y + attackerType.dimensionDown(), targetPosition.x - targetType.dimensionLeft() - 1, targetPosition.y - targetType.dimensionUp() - 1, targetPosition.x + targetType.dimensionRight() + 1, targetPosition.y + targetType.dimensionDown() + 1);
}


bool MyBot::goWithoutDamage(Unit u, Position target, int direction, int dangerGap)
{
	// 목적지 도달
	if (target.getApproxDistance(u->getPosition()) < 2 * TILE_SIZE)
		return true;

	bool isFlyer = u->isFlying() ? true : false;
	// configuration 가능 , 각도, Gap
	int angle = 20;

	int dangerPoint = 0;
	UnitInfo *dangerUnit = getDangerUnitNPoint(u->getPosition(), &dangerPoint, isFlyer);

	if (dangerPoint > dangerGap || dangerUnit == nullptr)
	{
		CommandUtil::move(u, target);
		bw->drawCircleMap(getDirectionDistancePosition(u->getPosition(), target, 2 * TILE_SIZE), 2, Colors::Red, true);
		return true;
	}

	// 각도 체크
	Position vector1 = dangerUnit->pos() - u->getPosition();
	Position vector2 = target - u->getPosition();
	int inner = (vector1.x * vector2.x) + (vector1.y * vector2.y);

	if (inner < 0)
	{
		CommandUtil::move(u, target);
		bw->drawCircleMap(getDirectionDistancePosition(u->getPosition(), target, 2 * TILE_SIZE), 2, Colors::Red, true);
		return true;
	}

	int weaponRange = weaponRange = isFlyer ? E->weaponMaxRange(dangerUnit->type().airWeapon()) : E->weaponMaxRange(dangerUnit->type().groundWeapon());

	Position back = getDirectionDistancePosition(dangerUnit->pos(), u->getPosition(), weaponRange + dangerGap);
	Position movePos = getCirclePosFromPosByDegree(dangerUnit->pos(), back, angle * direction);

	if (movePos.isValid() == false)
		return false;

	if (isFlyer == false && bw->isWalkable((WalkPosition)movePos) == false)
		return false;

	u->move(movePos);
	bw->drawCircleMap(movePos, 2, Colors::Red, true);
	return true;
}

void MyBot::kiting(UnitInfo *attacker, UnitInfo *target, int distance, int threshold)
{
	int backDistance = 3;
	int weapon_range = attacker->type().groundWeapon().maxRange();
	int en_weapon_range = target->type().groundWeapon().maxRange();
	int distToTarget = attacker->pos().getApproxDistance(target->pos());

	if (target->type().isWorker())
		backDistance = 2;

	if (weapon_range - en_weapon_range <= 2 * TILE_SIZE) // 히드라 마린
		threshold = weapon_range - en_weapon_range + TILE_SIZE;

	if (distance > threshold)
	{
		if (attacker->unit()->getGroundWeaponCooldown() == 0)
			CommandUtil::attackUnit(attacker->unit(), target->unit());

		if (attacker->posChange(target) == PosChange::Farther)
			attacker->unit()->move((attacker->pos() + target->vPos()) / 2);
		else if (!target->type().isWorker() && distToTarget < (weapon_range + en_weapon_range))
			moveBackPostion(attacker, target->pos(), backDistance * TILE_SIZE);
	}
	else
		moveBackPostion(attacker, target->pos(), backDistance * TILE_SIZE);
}

void MyBot::attackFirstkiting(UnitInfo *attacker, UnitInfo *target, int distance, int threshold)
{
	int backDistance = 2;

	if (attacker->unit()->getGroundWeaponCooldown() == 0)
		CommandUtil::attackUnit(attacker->unit(), target->unit());
	else if (distance < threshold)
		moveBackPostion(attacker, target->pos(), backDistance * TILE_SIZE);
}

void MyBot::pControl(UnitInfo *attacker, UnitInfo *target)
{
	if (attacker->unit()->getGroundWeaponCooldown() == 0)
	{
		if (attacker->pos().getApproxDistance(target->pos()) > 6 * TILE_SIZE)
			attacker->unit()->attack(target->unit());
		else
		{
			Position patrolPos = getDirectionDistancePosition(attacker->pos(), target->pos(), attacker->pos().getApproxDistance(target->pos()) + 2 * TILE_SIZE);

			if ((TIME - attacker->unit()->getLastCommandFrame() < 24) && attacker->unit()->getLastCommand().getType() == UnitCommandTypes::Patrol)
				return;

			attacker->unit()->patrol(patrolPos);
		}
	}
	else
	{
		if (attacker->posChange(target) == PosChange::Farther && attacker->pos().getApproxDistance(target->vPos()) > 4 * TILE_SIZE)
			attacker->unit()->move((attacker->pos() + target->vPos()) / 2);
		else {
			if (!target->type().isWorker() || attacker->pos().getApproxDistance(target->vPos()) < 2 * TILE_SIZE)
				moveBackPostion(attacker, target->pos(), 3 * TILE_SIZE);
		}
	}
}

UnitInfo *MyBot::getGroundWeakTargetInRange(UnitInfo *attacker, bool worker)
{
	uList enemy;

	if (worker)
		enemy = INFO.getUnitsInRadius(E, attacker->pos(), (int)(1.5 * S->weaponMaxRange(attacker->type().groundWeapon())), true, false, true);
	else
		enemy = INFO.getUnitsInRadius(E, attacker->pos(), (int)(1.5 * S->weaponMaxRange(attacker->type().groundWeapon())), true, false, false);

	int hp = INT_MAX;
	UnitInfo *u = nullptr;

	for (auto eu : enemy)
	{
		if (eu->type() == UnitTypes::Zerg_Egg || eu->type() == UnitTypes::Zerg_Larva || eu->type() == UnitTypes::Protoss_Interceptor ||
				eu->type() == UnitTypes::Protoss_Scarab || eu->type() == UnitTypes::Zerg_Broodling)
			continue;

		if (hp > eu->hp())
		{
			hp = eu->hp();
			u = eu;
		}
	}

	return u;
}

vector<int> MyBot::getNearObstacle(UnitInfo *uInfo, int directCnt, bool resource)
{
	vector<int> altitues;

	Position standard(TILE_SIZE, 0); // 3시 방향

	int angle = 360 / directCnt;
	int checkStep = WALKPOSITION_SCALE;
	int totalAngle = 0;
	int max_distance = 320;
	bool hasMineralOrGas = false;

	while (totalAngle < 360)
	{
		Position anglePos = getCirclePosFromPosByDegree(uInfo->pos(), uInfo->pos() + standard, totalAngle);

		int i = 1;

		for (i = checkStep; i  <= max_distance; i += checkStep)
		{
			Position direcPos = getDirectionDistancePosition(uInfo->pos(), anglePos, i);
			WalkPosition wPos = WalkPosition(direcPos);
			//bw->drawCircleMap(direcPos, 2, Colors::Blue, true);
			TilePosition tPos(direcPos);

			if (resource) {
				hasMineralOrGas = bw->getUnitsOnTile(tPos.x, tPos.y, Filter::IsResourceContainer).size() == 0 ? false : true;
			}

			if (hasMineralOrGas) {
				bw->drawCircleMap(direcPos, 2, Colors::Blue, true);
			}

			if (bw->isWalkable(wPos) == false || hasMineralOrGas)
			{
				altitues.push_back((i - checkStep));
				break;
			}
		}

		if (i > max_distance)
			altitues.push_back(max_distance);

		totalAngle += angle;
	}

	return altitues;
}


// Get degree of dangerous of each direction
vector<pair<double, double>> MyBot::getRadianAndDistanceFromEnemy(UnitInfo *uInfo, int directCnt)
{
	vector<pair<double, double>> danger_degree;
	int totalAngle = 0;
	int max_distance = 10 * TILE_SIZE;

	uList enemy = INFO.getUnitsInRadius(E, uInfo->pos(), max_distance, true, true, true, true, true);

	for (auto e : enemy) {
		double radian = getRadian2(uInfo->pos(), e->pos()); // To transform into not rotated coordinates, add M_PI / 2
		danger_degree.push_back(pair<double, double>(radian, uInfo->pos().getDistance(e->pos()))); // push radian, distance
	}

	return danger_degree;
}

vector<int> MyBot::getEnemiesInAngle(UnitInfo *uInfo, uList enemies, int directCnt, int range)
{
	int INFO_SIZE = 4;
	vector<int> enemyAngle;
	vector<double> thresholdMap;

	for (int i = 0; i < directCnt; i++) {
		for (int j = 0; j < INFO_SIZE; j++)
			enemyAngle.push_back(0);

		thresholdMap.push_back(M_PI * (i * 2) / directCnt);
		thresholdMap.push_back(M_PI * (i * 2 + 1) / directCnt);
	}

	for (auto z : enemies)
	{
		if (!z->unit()->exists() || z->id() == uInfo->id())
			continue;

		double dist = z->unit()->getDistance(uInfo->unit());

		if (dist > range)
			continue;

		double rad = getRadian(uInfo->pos(), z->pos()) * -1 + 2 * M_PI;

		word index = 0;

		for (int i = 1; i < directCnt; i++)
		{
			if (thresholdMap.at(i * 2 - 1) <= rad && rad < thresholdMap.at(i * 2 + 1)) {
				index = i;
				break;
			}
		}

		//if (z->pos().getDistance(uInfo->pos()) < 3 * TILE_SIZE)
		enemyAngle[index * INFO_SIZE]++;
		//else
		//    enemyAngle[index * 3 + 1]++;

		if (enemyAngle[index * INFO_SIZE + 1] < 7 * TILE_SIZE - dist) {
			enemyAngle[index * INFO_SIZE + 1] = (int)(7 * TILE_SIZE - dist);
			enemyAngle[index * INFO_SIZE + 2] = z->hp();
		}
	}

	return enemyAngle;
}
