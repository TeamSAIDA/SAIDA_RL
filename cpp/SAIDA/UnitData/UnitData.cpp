/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "../InformationManager.h"
#include "UnitData.h"

using namespace MyBot;

UnitData::UnitData()
{
}

UnitData::~UnitData()
{
}
///////// get UnitInfo Vector
vector<UnitInfo *> &UnitData::getUnitVector(UnitType uType)
{
	if (unitTypeMap.find(uType) == unitTypeMap.end())
	{
		vector<UnitInfo *> newUnitType;
		unitTypeMap[uType] = newUnitType;
	}

	return unitTypeMap[uType];
}

vector<UnitInfo *> &UnitData::getBuildingVector(UnitType uType)
{
	if (buildingTypeMap.find(uType) == buildingTypeMap.end())
	{
		vector<UnitInfo *> newUnitType;
		buildingTypeMap[uType] = newUnitType;
	}

	return buildingTypeMap[uType];
}
int UnitData::getCompletedCount(UnitType uType)
{
	if (completedCount.find(uType) == completedCount.end())
		completedCount[uType] = 0;

	return completedCount[uType];
}
int UnitData::getDestroyedCount(UnitType uType)
{
	if (destroyedCount.find(uType) == destroyedCount.end())
		destroyedCount[uType] = 0;

	return destroyedCount[uType];
}
map<UnitType, int> UnitData::getDestroyedCountMap() {
	return destroyedCount;
}
int UnitData::getAllCount(UnitType uType)
{
	if (allCount.find(uType) == allCount.end())
		allCount[uType] = 0;

	return allCount[uType];
}
void UnitData::increaseCompleteUnits(UnitType uType)
{
	UnitType type = getUnitTypeDB(uType);

	if (completedCount.find(type) == completedCount.end())
		completedCount[type] = 0;

	completedCount[type]++;
}
void UnitData::increaseDestroyUnits(UnitType uType)
{
	UnitType type = getUnitTypeDB(uType);

	if (destroyedCount.find(type) == destroyedCount.end())
		destroyedCount[type] = 0;

	destroyedCount[type]++;
}
void UnitData::increaseCreateUnits(UnitType uType)
{
	UnitType type = getUnitTypeDB(uType);

	if (allCount.find(type) == allCount.end())
		allCount[type] = 0;

	allCount[type]++;
}
void UnitData::decreaseCompleteUnits(UnitType uType)
{
	UnitType type = getUnitTypeDB(uType);

	completedCount[type]--;
}
void UnitData::decreaseCreateUnits(UnitType uType)
{
	UnitType type = getUnitTypeDB(uType);

	allCount[type]--;
}
//////// add UnitInfo
bool UnitData::addUnitNBuilding(Unit u)
{
	UnitType type = getUnitTypeDB(u->getType());

	map<Unit, UnitInfo *> &unitMap = type.isBuilding() ? getAllBuildings() : getAllUnits();
	vector<UnitInfo *> &unitVector = type.isBuilding() ? getBuildingVector(type) : getUnitVector(type);

	if (unitMap.find(u) == unitMap.end())
	{
		UnitInfo *pUnit = new UnitInfo(u);

		unitMap[u] = pUnit;
		unitVector.push_back(pUnit);

		return true;
	}

	return false;
}

////////////// remove UnitInfo
void UnitData::removeUnitNBuilding(Unit u)
{
	UnitType type = getUnitTypeDB(u->getType());

	map<Unit, UnitInfo *> &unitMap = type.isBuilding() ? getAllBuildings() : getAllUnits();
	vector<UnitInfo *> &unitVector = type.isBuilding() ? getBuildingVector(type) : getUnitVector(type);

	if (unitMap.find(u) != unitMap.end())
	{
		auto del_unit = find_if(unitVector.begin(), unitVector.end(), [u](UnitInfo * up) {
			return up->unit() == u;
		});

		if (del_unit != unitVector.end()) {
			BWEM::utils::fast_erase(unitVector, distance(unitVector.begin(), del_unit));
		}
		else {
			cout << "remove Unit Error" << endl;
		}

		delete unitMap[u];
		unitMap.erase(u);

		// Count를 -- 해준다.
		if (u->getPlayer() == S)
		{
			if (u->isCompleted())
				decreaseCompleteUnits(type);

			decreaseCreateUnits(type);
		}

		increaseDestroyUnits(type);
	}
}

void UnitData::clearUnitNBuilding() {

	for (auto u : allUnits)
		delete u.second;

	for (auto b : allBuildings)
		delete b.second;

	unitTypeMap.clear();
	buildingTypeMap.clear();
	allSpells.clear();
	completedCount.clear();
	destroyedCount.clear();
	allCount.clear();
	allUnits.clear();
	allBuildings.clear();
}

void UnitData::initializeAllInfo() {
	for (auto &u : allUnits)
		u.second->initFrame();

	for (auto &u : allBuildings)
		u.second->initFrame();
}

//////////// update UnitInfo
void UnitData::updateAllInfo()
{
	for (auto &u : allUnits)
		u.second->Update();

	for (auto &u : allBuildings)
		u.second->Update();
}

UnitType UnitData::getUnitTypeDB(UnitType uType)
{
	if (uType == Terran_Siege_Tank_Siege_Mode)
		return Terran_Siege_Tank_Tank_Mode;

	if (uType == Zerg_Lurker_Egg)
		return Zerg_Lurker;

	return uType;
}