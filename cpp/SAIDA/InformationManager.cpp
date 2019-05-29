/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "InformationManager.h"

using namespace MyBot;

InformationManager &InformationManager::Instance()
{
	static InformationManager instance;
	return instance;
}

void MyBot::InformationManager::initialize()
{
	mapPlayerLimit = bw->getStartLocations().size();

	selfPlayer = S;
	enemyPlayer = E;

	selfRace = S->getRace();
	enemyRace = E->getRace();
	enemySelectRace = E->getRace();

	_unitData[S] = UnitData();
	_unitData[E] = UnitData();
}

void InformationManager::updateManager()
{
	updateUnitsInfo();
}

void InformationManager::updateUnitsInfo()
{
	setUpgradeLevel();
	_unitData[selfPlayer].initializeAllInfo();
	_unitData[enemyPlayer].updateAllInfo();
	_unitData[selfPlayer].updateAllInfo();
}

void InformationManager::onUnitCreate(Unit unit)
{
	if (unit->getType().isNeutral() || unit->getPlayer() == E)
		return;

	// 건물은 Create 시점부터 DB에 저장
	if (unit->getType().isBuilding())
		_unitData[S].addUnitNBuilding(unit);

	_unitData[S].increaseCreateUnits(unit->getType());
}

void InformationManager::onUnitShow(Unit unit)
{
	if (unit->getType().isNeutral())
		return;

	_unitData[E].addUnitNBuilding(unit);

	if (enemyRace == Races::Unknown) {
		enemyRace = unit->getType().getRace();
	}
}

void InformationManager::onUnitComplete(Unit unit)
{
	if (unit->getType().isNeutral())
		return;

	if (unit->getPlayer() == S)
	{
		if (_unitData[S].addUnitNBuilding(unit)) // 새로 추가되는 경우만
			_unitData[S].increaseCompleteUnits(unit->getType());
		else if (unit->getType().isBuilding()) { // 건물은 Create 후 추가되므로 1번만 호출됨
			_unitData[S].increaseCompleteUnits(unit->getType());
		}
	}
	else if (unit->getPlayer() == E) // 적군의 경우 Show에서 추가 되었는지 안된지를 알수 없음.
	{
		_unitData[E].addUnitNBuilding(unit);

		if (enemyRace == Races::Unknown) {
			enemyRace = unit->getType().getRace();
		}
	}
}

// 유닛이 파괴/사망한 경우, 해당 유닛 정보를 삭제한다
void InformationManager::onUnitDestroy(Unit unit)
{
	if (unit == nullptr)
		return;

	if (unit->getType().isNeutral())
		return;

	_unitData[unit->getPlayer()].removeUnitNBuilding(unit);
}

void InformationManager::clearUnitNBuilding() {
	for (auto p : bw->getPlayers())
		_unitData[p].clearUnitNBuilding();
}

UnitType InformationManager::getBasicCombatUnitType(Race race)
{
	if (race == Races::None) {
		race = selfRace;
	}

	if (race == Races::Protoss) {
		return Protoss_Zealot;
	}
	else if (race == Races::Terran) {
		return Terran_Marine;
	}
	else if (race == Races::Zerg) {
		return Zerg_Zergling;
	}
	else {
		return None;
	}
}

UnitType InformationManager::getAdvancedCombatUnitType(Race race)
{
	if (race == Races::None) {
		race = selfRace;
	}

	if (race == Races::Protoss) {
		return Protoss_Dragoon;
	}
	else if (race == Races::Terran) {
		return Terran_Goliath;
	}
	else if (race == Races::Zerg) {
		return Zerg_Hydralisk;
	}
	else {
		return None;
	}
}

UnitType InformationManager::getBasicCombatBuildingType(Race race)
{
	if (race == Races::None) {
		race = selfRace;
	}

	if (race == Races::Protoss) {
		return Protoss_Gateway;
	}
	else if (race == Races::Terran) {
		return Terran_Barracks;
	}
	else if (race == Races::Zerg) {
		return Zerg_Hatchery;
	}
	else {
		return None;
	}
}

UnitType InformationManager::getObserverUnitType(Race race)
{
	if (race == Races::None) {
		race = selfRace;
	}

	if (race == Races::Protoss) {
		return Protoss_Observer;
	}
	else if (race == Races::Terran) {
		return Terran_Science_Vessel;
	}
	else if (race == Races::Zerg) {
		return Zerg_Overlord;
	}
	else {
		return None;
	}
}

UnitType	InformationManager::getBasicResourceDepotBuildingType(Race race)
{
	if (race == Races::None) {
		race = selfRace;
	}

	if (race == Races::Protoss) {
		return Protoss_Nexus;
	}
	else if (race == Races::Terran) {
		return Terran_Command_Center;
	}
	else if (race == Races::Zerg) {
		return Zerg_Hatchery;
	}
	else {
		return None;
	}
}
UnitType InformationManager::getRefineryBuildingType(Race race)
{
	if (race == Races::None) {
		race = selfRace;
	}

	if (race == Races::Protoss) {
		return Protoss_Assimilator;
	}
	else if (race == Races::Terran) {
		return Terran_Refinery;
	}
	else if (race == Races::Zerg) {
		return Zerg_Extractor;
	}
	else {
		return None;
	}

}

UnitType	InformationManager::getWorkerType(Race race)
{
	if (race == Races::None) {
		race = selfRace;
	}

	if (race == Races::Protoss) {
		return Protoss_Probe;
	}
	else if (race == Races::Terran) {
		return Terran_SCV;
	}
	else if (race == Races::Zerg) {
		return Zerg_Drone;
	}
	else {
		return None;
	}
}

UnitType InformationManager::getBasicSupplyProviderUnitType(Race race)
{
	if (race == Races::None) {
		race = selfRace;
	}

	if (race == Races::Protoss) {
		return Protoss_Pylon;
	}
	else if (race == Races::Terran) {
		return Terran_Supply_Depot;
	}
	else if (race == Races::Zerg) {
		return Zerg_Overlord;
	}
	else {
		return None;
	}
}

UnitType InformationManager::getAdvancedDefenseBuildingType(Race race, bool isAirDefense)
{
	if (race == Races::None) {
		race = selfRace;
	}

	if (race == Races::Protoss) {
		return Protoss_Photon_Cannon;
	}
	else if (race == Races::Terran) {
		return isAirDefense ? Terran_Missile_Turret : Terran_Bunker;
	}
	else if (race == Races::Zerg) {
		return isAirDefense ? Zerg_Spore_Colony : Zerg_Sunken_Colony;
	}
	else {
		return None;
	}
}

///////////////////////////////////////////
// 공통 Function 구현
///////////////////////////////////////////

// UnitData에서 UnitInfo를 찾아 Return함.
UnitInfo *InformationManager::getUnitInfo(Unit unit, Player p)
{
	if (unit == nullptr)
		return nullptr;

	if (unit->getType() == UnitTypes::Unknown) {
		if (_unitData[p].getAllBuildings().find(unit) != _unitData[p].getAllBuildings().end())
			return _unitData[p].getAllBuildings()[unit];

		if (_unitData[p].getAllUnits().find(unit) != _unitData[p].getAllUnits().end())
			return _unitData[p].getAllUnits()[unit];

		return nullptr;
	}

	uMap &AllUnitMap = unit->getType().isBuilding() ? _unitData[p].getAllBuildings() : _unitData[p].getAllUnits();

	if (AllUnitMap.find(unit) != AllUnitMap.end())
		return AllUnitMap[unit];
	else
		return nullptr;
}


int InformationManager::getCompletedCount(UnitType t, Player p)
{
	if (p == S)
		return _unitData[S].getCompletedCount(t);
	else
	{
		int compleCnt = 0;

		if (t.isBuilding()) {
			for (auto u : _unitData[E].getBuildingVector(t)) {
				if (u->isComplete())
					compleCnt++;
			}
		}
		else {
			for (auto u : _unitData[E].getUnitVector(t)) {
				if (u->isComplete())
					compleCnt++;
			}
		}

		return compleCnt;
	}
}


int InformationManager::getDestroyedCount(UnitType t, Player p)
{
	return _unitData[p].getDestroyedCount(t);
}


map<UnitType, int> InformationManager::getDestroyedCountMap(Player p)
{
	return _unitData[p].getDestroyedCountMap();
}


int InformationManager::getAllCount(UnitType t, Player p)
{
	if (p == S)
		return _unitData[S].getAllCount(t);
	else
	{
		if (t.isBuilding())
			return _unitData[E].getBuildingVector(t).size();
		else
			return _unitData[E].getUnitVector(t).size();
	}
}

// getUnitsInRadius
// pos에서 radius(pixel) 만큼 안에 있는 Unit을 가져온다.
// 필수 input : Player
// 옵션 input :
// pos, radius( 입력하지 않는 경우 모든 유닛을 가져온다. )
// ground (지상 유닛, 일꾼 포함) , air ( 공중 유닛 ), worker (일꾼 포함 여부, 참고로 일꾼만은 가져올 수 없음)
// hide( 현재 맵에서 없어진 유닛까지 가져올때 )
uList InformationManager::getUnitsInRadius(Player p, Position pos, int radius, bool ground, bool air, bool worker, bool hide, bool groundDistance)
{
	uList units;

	uMap allUnits = _unitData[p].getAllUnits();

	for (auto &u : allUnits)
	{
		if (u.second->type() != Zerg_Lurker && hide == false && u.second->isHide())
			continue;

		// 일단 Mine은 Skip
		if (u.second->type() == Zerg_Egg || u.second->type() == Zerg_Larva || u.second->type() == Protoss_Interceptor ||
				u.second->type() == Protoss_Scarab || u.second->type() == Terran_Vulture_Spider_Mine)
			continue;

		if (air == false && u.second->getLift())
			continue;

		if (ground == false && !u.second->getLift())
			continue;

		if (worker == false && u.second->type().isWorker())
			continue;

		if (radius)
		{
			if (u.second->pos() == Positions::Unknown)
				continue;

			if (groundDistance)
			{
				int dist = 0;
				theMap.GetPath(pos, u.second->pos(), &dist);

				if (dist < 0 || dist > radius)
					continue;

				units.push_back(u.second);
			}
			else
			{
				Position newPos = pos - u.second->pos();

				if (abs(newPos.x) > radius || abs(newPos.y) > radius)
					continue;

				if (abs(newPos.x) + abs(newPos.y) <= radius)
					units.push_back(u.second);
				else
				{
					if ((newPos.x * newPos.x) + (newPos.y * newPos.y) <= radius * radius)
						units.push_back(u.second);
				}
			}
		}
		else {
			units.push_back(u.second);
		}

	}

	return units;
}

uList InformationManager::getUnitsInRectangle(Player p, Position leftTop, Position rightDown, bool ground, bool air, bool worker, bool hide)
{
	uList units;

	uMap allUnits = _unitData[p].getAllUnits();

	for (auto &u : allUnits)
	{
		if (u.second->type() != Zerg_Lurker && hide == false && u.second->isHide())
			continue;

		if (u.second->type() == Zerg_Egg || u.second->type() == Zerg_Larva || u.second->type() == Protoss_Interceptor ||
				u.second->type() == Protoss_Scarab || u.second->type() == Terran_Vulture_Spider_Mine)
			continue;

		if (air == false && u.second->getLift())
			continue;

		if (ground == false && !u.second->getLift())
			continue;

		if (worker == false && u.second->type().isWorker())
			continue;

		if (u.second->pos() == Positions::Unknown)
			continue;

		int Threshold_L = leftTop.x;
		int Threshold_R = rightDown.x;
		int Threshold_T = leftTop.y;
		int Threshold_D = rightDown.y;

		//		if (u.second->unit()->getTop() > Threshold_D || u.second->unit()->getBottom() < Threshold_T || u.second->unit()->getLeft() > Threshold_R || u.second->unit()->getRight() < Threshold_L)
		if (u.second->pos().y > Threshold_D || u.second->pos().y < Threshold_T || u.second->pos().x > Threshold_R || u.second->pos().x  < Threshold_L)
			continue;

		units.push_back(u.second);
	}

	return units;
}

uList InformationManager::getBuildingsInRadius(Player p, Position pos, int radius, bool ground, bool air, bool hide, bool groundDistance)
{
	uList buildings;

	uMap allBuildings = _unitData[p].getAllBuildings();

	for (auto &u : allBuildings)
	{
		if (hide == false && u.second->isHide())
			continue;

		if (air == false && u.second->getLift())
			continue;

		if (ground == false && !u.second->getLift())
			continue;

		if (radius)
		{
			if (u.second->pos() == Positions::Unknown)
				continue;

			if (groundDistance)
			{
				int dist = 0;
				theMap.GetPath(pos, u.second->pos(), &dist);

				if (dist < 0 || dist > radius)
					continue;

				buildings.push_back(u.second);
			}
			else
			{
				Position newPos = pos - u.second->pos();

				if (abs(newPos.x) > radius || abs(newPos.y) > radius)
					continue;

				if (abs(newPos.x) + abs(newPos.y) <= radius)
					buildings.push_back(u.second);
				else
				{
					if ((newPos.x * newPos.x) + (newPos.y * newPos.y) <= radius * radius)
						buildings.push_back(u.second);
				}
			}
		}
		else
		{
			buildings.push_back(u.second);
		}
	}

	return buildings;
}

uList InformationManager::getBuildingsInRectangle(Player p, Position leftTop, Position rightDown, bool ground, bool air, bool hide)
{
	uList buildings;

	uMap allBuildings = _unitData[p].getAllBuildings();

	for (auto &u : allBuildings)
	{
		if (u.second->pos() == Positions::Unknown)
			continue;

		if (hide == false && u.second->isHide())
			continue;

		if (air == false && u.second->getLift())
			continue;

		if (ground == false && !u.second->getLift())
			continue;

		int Threshold_L = leftTop.x;
		int Threshold_R = rightDown.x;
		int Threshold_T = leftTop.y;
		int Threshold_D = rightDown.y;

		if (u.second->unit()->getTop() >= Threshold_D || u.second->unit()->getBottom() <= Threshold_T || u.second->unit()->getLeft() >= Threshold_R || u.second->unit()->getRight() <= Threshold_L)
			continue;

		buildings.push_back(u.second);
	}

	return buildings;
}

uList InformationManager::getUnitsInArea(Player p, Position pos, bool ground, bool air, bool worker, bool hide)
{
	uList units;

	uMap allUnits = _unitData[p].getAllUnits();

	for (auto &u : allUnits)
	{
		if (hide == false && u.second->isHide())
			continue;

		if (u.second->type() == Zerg_Egg || u.second->type() == Zerg_Larva || u.second->type() == Protoss_Interceptor ||
				u.second->type() == Protoss_Scarab || u.second->type() == Terran_Vulture_Spider_Mine)
			continue;

		if (air == false && u.second->getLift())
			continue;

		if (ground == false && !u.second->getLift())
			continue;

		if (worker == false && u.second->type().isWorker())
			continue;

		if (u.second->pos() == Positions::Unknown )
			continue;

		if (isSameArea(pos, u.second->pos()))
			units.push_back(u.second);
	}

	return units;
}

uList InformationManager::getBuildingsInArea(Player p, Position pos, bool ground, bool air, bool hide)
{
	uList buildings;

	for (auto &u : _unitData[p].getAllBuildings())
	{
		if (hide == false && u.second->isHide())
			continue;

		if (air == false && u.second->getLift())
			continue;

		if (ground == false && !u.second->getLift())
			continue;

		if (u.second->pos() == Positions::Unknown)
			continue;

		if (isSameArea(pos, u.second->pos()))
			buildings.push_back(u.second);
	}

	return buildings;
}

uList InformationManager::getAllInRadius(Player p, Position pos, int radius, bool ground, bool air, bool hide, bool groundDist)
{
	uList units = getUnitsInRadius(p, pos, radius, ground, air, true, hide, groundDist);
	uList buildings = getBuildingsInRadius(p, pos, radius, ground, air, hide, groundDist);
	units.insert(units.end(), buildings.begin(), buildings.end());
	return units;
}

uList InformationManager::getAllInRectangle(Player p, Position leftTop, Position rightDown, bool ground, bool air, bool hide)
{
	uList units = getUnitsInRectangle(p, leftTop, rightDown, ground, air, true, hide);
	uList buildings = getBuildingsInRectangle(p, leftTop, rightDown, ground, air, hide);
	units.insert(units.end(), buildings.begin(), buildings.end());
	return units;
}

uList InformationManager::getTypeUnitsInRadius(UnitType t, Player p, Position pos, int radius, bool hide)
{
	uList units;

	for (auto u : _unitData[p].getUnitVector(t))
	{
		if (hide == false && u->isHide())
			continue;

		if (radius)
		{
			if (u->pos() == Positions::Unknown)
				continue;

			Position newPos = pos - u->pos();

			if (abs(newPos.x) > radius || abs(newPos.y) > radius)
				continue;

			if (abs(newPos.x) + abs(newPos.y) <= radius)
				units.push_back(u);
			else
			{
				if ((newPos.x * newPos.x) + (newPos.y * newPos.y) <= radius * radius)
					units.push_back(u);
			}
		}
		else
		{
			units.push_back(u);
		}
	}

	return units;
}
uList InformationManager::getTypeBuildingsInRadius(UnitType t, Player p, Position pos, int radius, bool incomplete, bool hide)
{
	uList buildings;

	for (auto u : _unitData[p].getBuildingVector(t))
	{
		if (hide == false && u->isHide())
			continue;

		if (incomplete == false && (!u->isComplete() || u->isMorphing()))
			continue;

		if (radius)
		{
			if (u->pos() == Positions::Unknown)
				continue;

			Position newPos = pos - u->pos();

			if (abs(newPos.x) > radius || abs(newPos.y) > radius)
				continue;

			if (abs(newPos.x) + abs(newPos.y) <= radius)
				buildings.push_back(u);
			else
			{
				if ((newPos.x * newPos.x) + (newPos.y * newPos.y) <= radius * radius)
					buildings.push_back(u);
			}
		}
		else
		{
			buildings.push_back(u);
		}
	}

	return buildings;
}

uList InformationManager::getTypeUnitsInRectangle(UnitType t, Player p, Position leftTop, Position rightDown, bool hide)
{
	uList units;

	for (auto u : _unitData[p].getUnitVector(t))
	{
		if (u->pos() == Positions::Unknown)
			continue;

		if (hide == false && u->isHide())
			continue;

		int Threshold_L = leftTop.x;
		int Threshold_R = rightDown.x;
		int Threshold_T = leftTop.y;
		int Threshold_D = rightDown.y;

		//		if (u.second->unit()->getTop() > Threshold_D || u.second->unit()->getBottom() < Threshold_T || u.second->unit()->getLeft() > Threshold_R || u.second->unit()->getRight() < Threshold_L)
		if (u->pos().y > Threshold_D || u->pos().y < Threshold_T || u->pos().x > Threshold_R || u->pos().x  < Threshold_L)
			continue;

		units.push_back(u);
	}

	return units;
}

uList InformationManager::getTypeBuildingsInRectangle(UnitType t, Player p, Position leftTop, Position rightDown, bool incomplete, bool hide)
{
	uList buildings;

	for (auto u : _unitData[p].getBuildingVector(t))
	{
		if (u->pos() == Positions::Unknown)
			continue;

		if (hide == false && u->isHide())
			continue;

		if (incomplete == false && (!u->isComplete() || u->isMorphing()))
			continue;

		int Threshold_L = leftTop.x;
		int Threshold_R = rightDown.x;
		int Threshold_T = leftTop.y;
		int Threshold_D = rightDown.y;

		if (u->unit()->getTop() >= Threshold_D || u->unit()->getBottom() <= Threshold_T || u->unit()->getLeft() >= Threshold_R || u->unit()->getRight() <= Threshold_L)
			continue;

		buildings.push_back(u);
	}

	return buildings;
}

uList InformationManager::getTypeUnitsInArea(UnitType t, Player p, Position pos, bool hide)
{
	uList units;

	for (auto u : _unitData[p].getUnitVector(t))
	{
		if (hide == false && u->isHide())
			continue;

		if (u->pos() == Positions::Unknown)
			continue;

		if (isSameArea(u->pos(), pos))
			units.push_back(u);
	}

	return units;
}

uList InformationManager::getTypeBuildingsInArea(UnitType t, Player p, Position pos, bool incomplete, bool hide)
{
	uList buildings;

	for (auto u : _unitData[p].getBuildingVector(t))
	{
		if (hide == false && u->isHide())
			continue;

		if (incomplete == false && (!u->isComplete() || u->isMorphing()))
			continue;

		if (u->pos() == Positions::Unknown)
			continue;

		if (isSameArea(u->pos(), pos))
			buildings.push_back(u);
	}

	return buildings;
}

uList InformationManager::getDefenceBuildingsInRadius(Player p, Position pos, int radius, bool incomplete, bool hide)
{
	UnitType t = p == S ? getAdvancedDefenseBuildingType(INFO.selfRace) : getAdvancedDefenseBuildingType(INFO.enemyRace);
	return getTypeBuildingsInRadius(t, p, pos, radius, incomplete, hide);
}

UnitInfo *InformationManager::getClosestUnit(Player p, Position pos, TypeKind kind, int radius, bool worker, bool hide, bool groundDistance, bool detectedOnly)
{
	UnitInfo *closest = nullptr;
	int closestDist = INT_MAX;

	if (kind == TypeKind::AllUnitKind || kind == TypeKind::AirUnitKind || kind == TypeKind::GroundUnitKind || kind == TypeKind::GroundCombatKind || kind == TypeKind::AllKind)
	{
		uMap allUnits = _unitData[p].getAllUnits();

		for (auto &u : allUnits)
		{
			if (u.second->type().isFlyer() && (kind == GroundUnitKind || kind == GroundCombatKind))
				continue;

			if (!u.second->getLift() && kind == AirUnitKind)
				continue;

			if (u.second->type().isWorker() && worker == false)
				continue;

			if (hide == false && u.second->isHide())
				continue;

			if (u.second->pos() == Positions::Unknown)
				continue;

			if (detectedOnly && !u.second->isHide() && !u.second->unit()->isDetected())
				continue;

			// Closest 유닛에서 제외하는 종류
			if (u.second->type() == Zerg_Egg || u.second->type() == Zerg_Larva || u.second->type() == Protoss_Interceptor ||
					u.second->type() == Protoss_Scarab || u.second->type() == Terran_Vulture_Spider_Mine)
				continue;

			Position newPos = pos - u.second->pos();

			if (groundDistance)
			{
				int dist = 0;
				theMap.GetPath(pos, u.second->pos(), &dist);

				if (radius && dist > radius)
					continue;

				if (dist > 0 && dist < closestDist)
				{
					closest = u.second;
					closestDist = dist;
				}
			}
			else
			{
				if (radius)
				{
					if (abs(newPos.x) > radius || abs(newPos.y) > radius)
						continue;

					if ((newPos.x * newPos.x) + (newPos.y * newPos.y) > radius * radius)
						continue;
				}

				if ((newPos.x * newPos.x + newPos.y * newPos.y) < closestDist)
				{
					closest = u.second;
					closestDist = (newPos.x * newPos.x + newPos.y * newPos.y);
				}
			}
		}
	}

	if (kind == TypeKind::BuildingKind || kind == TypeKind::AllDefenseBuildingKind || kind == TypeKind::AirDefenseBuildingKind || kind == TypeKind::GroundDefenseBuildingKind || kind == TypeKind::AllKind || kind == GroundCombatKind)
	{
		uMap allBuildings = _unitData[p].getAllBuildings();

		for (auto &u : allBuildings)
		{
			if (hide == false && u.second->isHide())
				continue;

			if (u.second->pos() == Positions::Unknown)
				continue;

			if (u.second->isMorphing() || !u.second->isComplete())
				continue;

			if (!u.second->type().airWeapon().targetsAir() && u.second->type() != Terran_Bunker && (kind == AirDefenseBuildingKind || kind == AllDefenseBuildingKind))
				continue;

			if (!u.second->type().groundWeapon().targetsGround() && u.second->type() != Terran_Bunker && (kind == GroundDefenseBuildingKind || kind == AllDefenseBuildingKind || kind == GroundCombatKind))
				continue;

			Position newPos = pos - u.second->pos();

			if (groundDistance)
			{
				int dist = 0;
				theMap.GetPath(pos, u.second->pos(), &dist);

				if (radius && dist > radius)
					continue;

				if (dist > 0 && dist < closestDist)
				{
					closest = u.second;
					closestDist = dist;
				}
			}
			else
			{
				if (radius)
				{
					if (abs(newPos.x) > radius || abs(newPos.y) > radius)
						continue;

					if ((newPos.x * newPos.x) + (newPos.y * newPos.y) > radius * radius)
						continue;
				}

				if ((newPos.x * newPos.x + newPos.y * newPos.y) < closestDist)
				{
					closest = u.second;
					closestDist = (newPos.x * newPos.x + newPos.y * newPos.y);
				}
			}
		}
	}

	return closest;
}

UnitInfo *InformationManager::getFarthestUnit(Player p, Position pos, TypeKind kind, int radius, bool worker, bool hide, bool groundDistance, bool detectedOnly)
{
	UnitInfo *closest = nullptr;
	int closestDist = -1;

	if (kind == TypeKind::AllUnitKind || kind == TypeKind::AirUnitKind || kind == TypeKind::GroundUnitKind || kind == TypeKind::GroundCombatKind || kind == TypeKind::AllKind)
	{
		uMap allUnits = _unitData[p].getAllUnits();

		for (auto &u : allUnits)
		{
			if (u.second->type().isFlyer() && (kind == GroundUnitKind || kind == GroundCombatKind))
				continue;

			if (!u.second->unit()->isFlying() && kind == AirUnitKind)
				continue;

			if (u.second->type().isWorker() && worker == false)
				continue;

			if (hide == false && u.second->isHide())
				continue;

			if (u.second->pos() == Positions::Unknown)
				continue;

			if (detectedOnly && !u.second->isHide() && !u.second->unit()->isDetected())
				continue;

			// Closest 유닛에서 제외하는 종류
			if (u.second->type() == Zerg_Egg || u.second->type() == Zerg_Larva || u.second->type() == Protoss_Interceptor ||
					u.second->type() == Protoss_Scarab || u.second->type() == Terran_Vulture_Spider_Mine)
				continue;

			Position newPos = pos - u.second->pos();

			if (groundDistance)
			{
				int dist = 0;
				theMap.GetPath(pos, u.second->pos(), &dist);

				if (radius && dist > radius)
					continue;

				if (dist > 0 && dist > closestDist)
				{
					closest = u.second;
					closestDist = dist;
				}
			}
			else
			{
				if (radius)
				{
					if (abs(newPos.x) > radius || abs(newPos.y) > radius)
						continue;

					if ((newPos.x * newPos.x) + (newPos.y * newPos.y) > radius * radius)
						continue;
				}

				if ((newPos.x * newPos.x + newPos.y * newPos.y) > closestDist)
				{
					closest = u.second;
					closestDist = (newPos.x * newPos.x + newPos.y * newPos.y);
				}
			}
		}
	}

	if (kind == TypeKind::BuildingKind || kind == TypeKind::AllDefenseBuildingKind || kind == TypeKind::AirDefenseBuildingKind || kind == TypeKind::GroundDefenseBuildingKind || kind == TypeKind::AllKind || kind == GroundCombatKind)
	{
		for (auto &u : _unitData[p].getAllBuildings())
		{
			if (hide == false && u.second->isHide())
				continue;

			if (u.second->pos() == Positions::Unknown)
				continue;

			if (!u.second->unit()->getType().airWeapon().targetsAir() && (kind == AirDefenseBuildingKind || kind == AllDefenseBuildingKind))
				continue;

			if (!u.second->unit()->getType().groundWeapon().targetsGround() && (kind == GroundDefenseBuildingKind || kind == AllDefenseBuildingKind || kind == GroundCombatKind))
				continue;

			Position newPos = pos - u.second->pos();

			if (groundDistance)
			{
				int dist = 0;
				theMap.GetPath(pos, u.second->pos(), &dist);

				if (radius && dist > radius)
					continue;

				if (dist > 0 && dist > closestDist)
				{
					closest = u.second;
					closestDist = dist;
				}
			}
			else
			{
				if (radius)
				{
					if (abs(newPos.x) > radius || abs(newPos.y) > radius)
						continue;

					if ((newPos.x * newPos.x) + (newPos.y * newPos.y) > radius * radius)
						continue;
				}

				if ((newPos.x * newPos.x + newPos.y * newPos.y) > closestDist)
				{
					closest = u.second;
					closestDist = (newPos.x * newPos.x + newPos.y * newPos.y);
				}
			}
		}
	}

	return closest;
}

UnitInfo *InformationManager::getClosestTypeUnit(Player p, Position pos, UnitType type, int radius, bool hide, bool groundDistance, bool detectedOnly)
{
	if (!pos.isValid())
		return nullptr;

	UnitInfo *closest = nullptr;
	int closestDist = INT_MAX;

	uList &unitVector = type.isBuilding() ? _unitData[p].getBuildingVector(type) : _unitData[p].getUnitVector(type);

	for (auto &u : unitVector)
	{
		if (hide == false && u->isHide())
			continue;

		if (u->pos() == Positions::Unknown)
			continue;

		if (detectedOnly && !u->isHide() && !u->unit()->isDetected())
			continue;

		Position newPos = pos - u->pos();

		if (groundDistance)
		{
			int dist = 0;
			theMap.GetPath(pos, u->pos(), &dist);

			if (radius && dist > radius)
				continue;

			if (dist >= 0 && dist < closestDist)
			{
				closest = u;
				closestDist = dist;
			}
		}
		else
		{
			if (radius)
			{
				if (abs(newPos.x) > radius || abs(newPos.y) > radius)
					continue;

				if ((newPos.x * newPos.x) + (newPos.y * newPos.y) > radius * radius)
					continue;
			}

			if ((newPos.x * newPos.x + newPos.y * newPos.y) < closestDist)
			{
				closest = u;
				closestDist = (newPos.x * newPos.x + newPos.y * newPos.y);
			}
		}
	}

	return closest;
}

UnitInfo *InformationManager::getClosestTypeUnit(Player p, Position pos, vector<UnitType> &types, int radius, bool hide, bool groundDistance, bool detectedOnly)
{
	if (!pos.isValid())
		return nullptr;

	UnitInfo *closest = nullptr;
	int closestDist = INT_MAX;

	for (auto type : types)
	{
		uList &unitVector = type.isBuilding() ? _unitData[p].getBuildingVector(type) : _unitData[p].getUnitVector(type);

		for (auto u : unitVector)
		{
			if (hide == false && u->isHide())
				continue;

			if (u->pos() == Positions::Unknown)
				continue;

			if (detectedOnly && !u->isHide() && !u->unit()->isDetected())
				continue;

			Position newPos = pos - u->pos();

			if (groundDistance)
			{
				int dist = 0;
				theMap.GetPath(pos, u->pos(), &dist);

				if (radius && dist > radius)
					continue;

				if (dist >= 0 && dist < closestDist)
				{
					closest = u;
					closestDist = dist;
				}
			}
			else
			{
				if (radius)
				{
					if (abs(newPos.x) > radius || abs(newPos.y) > radius)
						continue;

					if ((newPos.x * newPos.x) + (newPos.y * newPos.y) > radius * radius)
						continue;
				}

				if ((newPos.x * newPos.x + newPos.y * newPos.y) < closestDist)
				{
					closest = u;
					closestDist = (newPos.x * newPos.x + newPos.y * newPos.y);
				}
			}
		}
	}

	return closest;
}

UnitInfo *InformationManager::getFarthestTypeUnit(Player p, Position pos, UnitType type, int radius, bool hide, bool groundDistance, bool detectedOnly)
{
	UnitInfo *farthest = nullptr;
	int farthestDist = 0;

	uList &unitVector = type.isBuilding() ? _unitData[p].getBuildingVector(type) : _unitData[p].getUnitVector(type);

	for (auto u : unitVector)
	{
		if (hide == false && u->isHide())
			continue;

		if (u->pos() == Positions::Unknown)
			continue;

		if (detectedOnly && !u->isHide() && !u->unit()->isDetected())
			continue;

		Position newPos = pos - u->pos();

		if (groundDistance)
		{
			int dist = 0;
			theMap.GetPath(pos, u->pos(), &dist);

			if (radius && dist > radius)
				continue;

			if (dist > farthestDist)
			{
				farthest = u;
				farthestDist = dist;
			}
		}
		else
		{
			if (radius)
			{
				if (abs(newPos.x) > radius || abs(newPos.y) > radius)
					continue;

				if ((newPos.x * newPos.x) + (newPos.y * newPos.y) > radius * radius)
					continue;
			}

			if ((newPos.x * newPos.x + newPos.y * newPos.y) > farthestDist)
			{
				farthest = u;
				farthestDist = (newPos.x * newPos.x + newPos.y * newPos.y);
			}
		}
	}

	return farthest;
}

bool InformationManager::hasResearched(TechType tech) {
	return researchedSet.find(tech) != researchedSet.end();
}

void InformationManager::setResearched(UnitType unitType) {
	if (unitType == Terran_Siege_Tank_Siege_Mode)
		researchedSet.insert(TechTypes::Tank_Siege_Mode);
	else if (unitType == Zerg_Lurker || unitType == Zerg_Lurker_Egg)
		researchedSet.insert(TechTypes::Lurker_Aspect);
	else if (unitType == Terran_Vulture_Spider_Mine)
		researchedSet.insert(TechTypes::Spider_Mines);
}

void InformationManager::setUpgradeLevel()
{
	if (INFO.enemyRace == Races::Unknown)
		return;

	if (upgradeList.empty())
	{
		// 종족 별로 필요한 upgrade Type을 추가 한다.
		if (INFO.enemyRace == Races::Terran)
		{
			upgradeList.push_back(UpgradeTypes::Charon_Boosters);
			upgradeList.push_back(UpgradeTypes::Ion_Thrusters);
			upgradeList.push_back(UpgradeTypes::U_238_Shells);
			upgradeList.push_back(UpgradeTypes::Terran_Vehicle_Weapons);
		}
		else if (INFO.enemyRace == Races::Zerg)
		{
			upgradeList.push_back(UpgradeTypes::Ventral_Sacs);
			upgradeList.push_back(UpgradeTypes::Muscular_Augments);
			upgradeList.push_back(UpgradeTypes::Grooved_Spines);
			upgradeList.push_back(UpgradeTypes::Metabolic_Boost);
		}
		else // Protoss
		{
			upgradeList.push_back(UpgradeTypes::Leg_Enhancements);
			upgradeList.push_back(UpgradeTypes::Singularity_Charge);
		}
	}

	for (word i = 0; i < upgradeList.size(); i++)
	{
		if (upgradeSet.find(upgradeList[i]) == upgradeSet.end())
			upgradeSet[upgradeList[i]] = 0;
		else
			upgradeSet[upgradeList[i]] = max(upgradeSet[upgradeList[i]], E->getUpgradeLevel(upgradeList[i]));
	}
}

int InformationManager::getUpgradeLevel(UpgradeType up)
{
	return upgradeSet[up];
}

vector<UnitType> InformationManager::getCombatTypes(Race race)
{
	vector<UnitType> unitTypes;

	if (race == Races::Terran)
	{
		unitTypes.push_back(Terran_Marine);
		unitTypes.push_back(Terran_Firebat);
		unitTypes.push_back(Terran_Medic);

		unitTypes.push_back(Terran_Vulture);
		unitTypes.push_back(Terran_Siege_Tank_Tank_Mode);
		unitTypes.push_back(Terran_Goliath);

		unitTypes.push_back(Terran_Wraith);
		unitTypes.push_back(Terran_Battlecruiser);
	}
	else if (race == Races::Zerg)
	{
		unitTypes.push_back(Zerg_Zergling);
		unitTypes.push_back(Zerg_Hydralisk);
		unitTypes.push_back(Zerg_Lurker);

		unitTypes.push_back(Zerg_Scourge);
		unitTypes.push_back(Zerg_Mutalisk);
		unitTypes.push_back(Zerg_Guardian);
		unitTypes.push_back(Zerg_Devourer);

		unitTypes.push_back(Zerg_Ultralisk);
		unitTypes.push_back(Zerg_Defiler);
	}
	else
	{
		unitTypes.push_back(Protoss_Zealot);
		unitTypes.push_back(Protoss_Dragoon);
		unitTypes.push_back(Protoss_Carrier);
		unitTypes.push_back(Protoss_Scout);

		unitTypes.push_back(Protoss_Arbiter);
		unitTypes.push_back(Protoss_Archon);
		unitTypes.push_back(Protoss_High_Templar);
		unitTypes.push_back(Protoss_Dark_Templar);
		unitTypes.push_back(Protoss_Reaver);
	}

	return unitTypes;
}
