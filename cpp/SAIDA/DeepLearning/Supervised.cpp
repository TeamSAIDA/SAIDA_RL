/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "Supervised.h"
#include <iostream>
#include <vector>

#define Num 128

using namespace MyBot;

Supervised &Supervised::Instance()
{
	static Supervised instance;
	return instance;
}

Supervised::Supervised()
{
	if (INFO.enemyRace == Races::Unknown || INFO.enemyRace == Races::None || INFO.enemyRace == Races::Random) return;

	unitListsTerran = { Terran_Firebat, Terran_Ghost, Terran_Goliath, Terran_Marine, Terran_Medic, Terran_SCV,
						Terran_Siege_Tank_Siege_Mode, Terran_Siege_Tank_Tank_Mode, Terran_Vulture, Terran_Vulture_Spider_Mine,
						Terran_Battlecruiser, Terran_Dropship, Terran_Nuclear_Missile, Terran_Science_Vessel, Terran_Valkyrie, Terran_Wraith
					  };

	unitListsProtoss = { Protoss_Archon, Protoss_Dark_Archon, Protoss_Dark_Templar, Protoss_Dragoon, Protoss_High_Templar, Protoss_Probe,
						 Protoss_Reaver, Protoss_Scarab, Protoss_Zealot, Protoss_Arbiter, Protoss_Carrier, Protoss_Corsair, Protoss_Interceptor,
						 Protoss_Observer, Protoss_Scout, Protoss_Shuttle
					   };

	unitListsZerg = { Zerg_Broodling, Zerg_Defiler, Zerg_Drone, Zerg_Egg, Zerg_Hydralisk, Zerg_Infested_Terran, Zerg_Larva, Zerg_Lurker,
					  Zerg_Lurker_Egg, Zerg_Ultralisk, Zerg_Zergling, Zerg_Cocoon, Zerg_Devourer, Zerg_Guardian,
					  Zerg_Mutalisk, Zerg_Overlord, Zerg_Queen, Zerg_Scourge
					};

	buildingListsTerran = { Terran_Academy, Terran_Armory, Terran_Barracks, Terran_Bunker, Terran_Command_Center, Terran_Engineering_Bay,
							Terran_Factory, Terran_Missile_Turret, Terran_Refinery, Terran_Science_Facility, Terran_Starport,
							Terran_Supply_Depot, Terran_Comsat_Station, Terran_Control_Tower, Terran_Covert_Ops, Terran_Machine_Shop,
							Terran_Nuclear_Silo, Terran_Physics_Lab
						  };

	buildingListsProtoss = { Protoss_Arbiter_Tribunal, Protoss_Assimilator, Protoss_Citadel_of_Adun, Protoss_Cybernetics_Core,
							 Protoss_Fleet_Beacon, Protoss_Forge, Protoss_Gateway, Protoss_Nexus, Protoss_Observatory,
							 Protoss_Photon_Cannon, Protoss_Pylon, Protoss_Robotics_Facility, Protoss_Robotics_Support_Bay,
							 Protoss_Shield_Battery, Protoss_Stargate, Protoss_Templar_Archives
						   };

	buildingListsZerg = { Zerg_Creep_Colony, Zerg_Defiler_Mound, Zerg_Evolution_Chamber, Zerg_Extractor,
						  Zerg_Greater_Spire, Zerg_Hatchery, Zerg_Hive, Zerg_Hydralisk_Den, Zerg_Infested_Command_Center,
						  Zerg_Lair, Zerg_Nydus_Canal, Zerg_Queens_Nest, Zerg_Spawning_Pool, Zerg_Spire,
						  Zerg_Spore_Colony, Zerg_Sunken_Colony, Zerg_Ultralisk_Cavern
						};

	makeMyFeatureMap();
	makeEnemyFeatureMap();
}
Supervised::~Supervised()
{
	clearAllFeatureMap();
}

void Supervised::update(Player p)
{
	if (p == S)
		updateMyFeatureMap();
	else if (p == E)
		updateEnemyFeatureMap();
}
int Supervised::getUnitAndBuildingCount(Player p)
{
	int count = 0;

	if (p->getRace() == Races::Protoss)
	{
		count = unitListsProtoss.size() + buildingListsProtoss.size();
	}
	else if (p->getRace() == Races::Terran)
	{
		count = unitListsTerran.size() + buildingListsTerran.size();
	}
	else if (p->getRace() == Races::Zerg)
	{
		count = unitListsZerg.size() + buildingListsZerg.size();
	}
	else
	{
		count = 0;
	}

	return count;
}
map<std::string, int **> Supervised::getFeatureMap(Player p)
{
	if (p == S)
		return featureMap_S;

	return featureMap_E;
}
std::list<UnitType> Supervised::getBuildingList(Player p)
{
	if (p->getRace() == Races::Protoss)
		return buildingListsProtoss;
	else if (p->getRace() == Races::Terran)
		return buildingListsTerran;

	return buildingListsZerg;
}
std::list<UnitType> Supervised::getUnitList(Player p)
{
	if (p->getRace() == Races::Protoss)
		return unitListsProtoss;
	else if (p->getRace() == Races::Terran)
		return unitListsTerran;

	return unitListsZerg;
}
void Supervised::clearAllFeatureMap()
{
	std::list<UnitType>::iterator it;
	std::list<UnitType> unitList;
	std::list<UnitType> buildingList;

	if (INFO.enemyRace == BWAPI::Races::Protoss)
	{
		unitList = unitListsProtoss;
		buildingList = buildingListsProtoss;
	}
	else if (INFO.enemyRace == BWAPI::Races::Terran)
	{
		unitList = unitListsTerran;
		buildingList = buildingListsTerran;
	}
	else if (INFO.enemyRace == BWAPI::Races::Zerg)
	{
		unitList = unitListsZerg;
		buildingList = buildingListsZerg;
	}

	for (it = unitListsTerran.begin(); it != unitListsTerran.end(); it++)
	{
		for (int i = 0; i < Num; i++)
		{
			if (featureMap_S[it->getName()][i] != nullptr)
			{
				free(featureMap_S[it->getName()][i]);
			}
		}

		if (featureMap_S[it->getName()] != nullptr)
		{
			free(featureMap_S[it->getName()]);
			featureMap_S[it->getName()] = nullptr;
		}
	}

	for (it = buildingListsTerran.begin(); it != buildingListsTerran.end(); it++)
	{
		for (int i = 0; i < Num; i++)
		{
			if (featureMap_S[it->getName()][i] != nullptr)
			{
				free(featureMap_S[it->getName()][i]);
			}
		}

		if (featureMap_S[it->getName()] != nullptr)
		{
			free(featureMap_S[it->getName()]);
			featureMap_S[it->getName()] = nullptr;
		}
	}

	// 상대
	for (it = unitList.begin(); it != unitList.end(); it++)
	{
		for (int i = 0; i < Num; i++)
		{
			if (featureMap_E[it->getName()][i] != nullptr)
			{
				free(featureMap_E[it->getName()][i]);
			}
		}

		if (featureMap_E[it->getName()] != nullptr)
		{
			free(featureMap_E[it->getName()]);
			featureMap_E[it->getName()] = nullptr;
		}
	}

	for (it = buildingList.begin(); it != buildingList.end(); it++)
	{
		for (int i = 0; i < Num; i++)
		{
			if (featureMap_E[it->getName()][i] != nullptr)
			{
				free(featureMap_E[it->getName()][i]);
			}
		}

		if (featureMap_E[it->getName()] != nullptr)
		{
			free(featureMap_E[it->getName()]);
			featureMap_E[it->getName()] = nullptr;
		}
	}
}
void Supervised::updateMyFeatureMap()
{
	int mapX, mapY, rateX, rateY;
	mapX = theMap.Size().x * TILE_SIZE;
	mapY = theMap.Size().y * TILE_SIZE;
	std::list<UnitType>::iterator it;
	std::list<UnitType> unitList = unitListsTerran;
	std::list<UnitType> buildingList = buildingListsTerran;

	for (it = unitList.begin(); it != unitList.end(); it++)
	{
		//map 정보 클리어
		for (int i = 0; i < Num; i++)
		{
			for (int j = 0; j < Num; j++)
			{
				featureMap_S[it->getName()][i][j] = 0;
			}
		}

		//유닛 정보 업데이트
		for (auto u : INFO.getTypeUnitsInRadius(*it, S))
		{
			rateX = u->pos().x * Num / mapX;
			rateY = u->pos().y * Num / mapY;
			featureMap_S[it->getName()][rateX][rateY]++;
		}
	}

	for (it = buildingList.begin(); it != buildingList.end(); it++)
	{
		//map 정보 클리어
		for (int i = 0; i < Num; i++)
		{
			for (int j = 0; j < Num; j++)
			{
				featureMap_S[it->getName()][i][j] = 0;
			}
		}

		//빌딩 정보 업데이트
		for (auto u : INFO.getTypeUnitsInRadius(*it, S))
		{
			rateX = u->pos().x * Num / mapX;
			rateY = u->pos().y * Num / mapY;
			featureMap_S[it->getName()][rateX][rateY]++;
		}
	}
}

void Supervised::updateEnemyFeatureMap()
{
	int mapX, mapY, rateX, rateY;
	mapX = theMap.Size().x * TILE_SIZE;
	mapY = theMap.Size().y * TILE_SIZE;
	std::list<UnitType>::iterator it;
	std::list<UnitType> unitList;
	std::list<UnitType> buildingList;

	if (INFO.enemyRace == BWAPI::Races::Protoss)
	{
		unitList = unitListsProtoss;
		buildingList = buildingListsProtoss;
	}
	else if (INFO.enemyRace == BWAPI::Races::Terran)
	{
		unitList = unitListsTerran;
		buildingList = buildingListsTerran;
	}
	else if (INFO.enemyRace == BWAPI::Races::Zerg)
	{
		unitList = unitListsZerg;
		buildingList = buildingListsZerg;
	}

	for (it = unitList.begin(); it != unitList.end(); it++)
	{
		//map 정보 클리어
		for (int i = 0; i < Num; i++)
		{
			for (int j = 0; j < Num; j++)
			{
				featureMap_E[it->getName()][i][j] = 0;
			}
		}

		//유닛 정보 업데이트
		for (auto u : INFO.getTypeUnitsInRadius(*it, E))
		{
			rateX = u->pos().x * Num / mapX;
			rateY = u->pos().y * Num / mapY;
			featureMap_E[it->getName()][rateX][rateY]++;
		}
	}

	for (it = buildingList.begin(); it != buildingList.end(); it++)
	{
		//map 정보 클리어
		for (int i = 0; i < Num; i++)
		{
			for (int j = 0; j < Num; j++)
			{
				featureMap_E[it->getName()][i][j] = 0;
			}
		}

		//빌딩 정보 업데이트
		for (auto u : INFO.getTypeUnitsInRadius(*it, E))
		{
			rateX = u->pos().x * Num / mapX;
			rateY = u->pos().y * Num / mapY;
			featureMap_E[it->getName()][rateX][rateY]++;
		}
	}
}

//내 피쳐맵 만들기
void Supervised::makeMyFeatureMap()
{
	std::list<UnitType>::iterator it;

	for (it = unitListsTerran.begin(); it != unitListsTerran.end(); it++)
	{
		featureMap_S[it->getName()] = nullptr;
		featureMap_S[it->getName()] = (int **)malloc(sizeof(int *) * Num);

		for (int i = 0; i < Num; i++)
		{
			featureMap_S[it->getName()][i] = (int *)malloc(sizeof(int) * Num);
		}

		for (int i = 0; i < Num; i++)
		{
			for (int j = 0; j < Num; j++)
			{
				featureMap_S[it->getName()][i][j] = 0;
			}
		}
	}

	for (it = buildingListsTerran.begin(); it != buildingListsTerran.end(); it++)
	{
		featureMap_S[it->getName()] = nullptr;
		featureMap_S[it->getName()] = (int **)malloc(sizeof(int *) * Num);

		for (int i = 0; i < Num; i++)
		{
			featureMap_S[it->getName()][i] = (int *)malloc(sizeof(int) * Num);
		}

		for (int i = 0; i < Num; i++)
		{
			for (int j = 0; j < Num; j++)
			{
				featureMap_S[it->getName()][i][j] = 0;
			}
		}
	}
}

//상대방 피쳐맵 만들기
void Supervised::makeEnemyFeatureMap()
{
	std::list<UnitType>::iterator it;
	std::list<UnitType> unitList;
	std::list<UnitType> buildingList;

	if (INFO.enemyRace == Races::Terran)
	{
		unitList = unitListsTerran;
		buildingList = buildingListsTerran;
	}
	else if (INFO.enemyRace == Races::Protoss)
	{
		unitList = unitListsProtoss;
		buildingList = buildingListsProtoss;
	}
	else if (INFO.enemyRace == Races::Zerg)
	{
		unitList = unitListsZerg;
		buildingList = buildingListsZerg;
	}

	for (it = unitList.begin(); it != unitList.end(); it++)
	{
		featureMap_E[it->getName()] = (int **)malloc(sizeof(int *) * Num);

		for (int i = 0; i < Num; i++)
		{
			featureMap_E[it->getName()][i] = (int *)malloc(sizeof(int) * Num);
		}

		for (int i = 0; i < Num; i++)
		{
			for (int j = 0; j < Num; j++)
			{
				featureMap_E[it->getName()][i][j] = 0;
			}
		}
	}

	for (it = buildingList.begin(); it != buildingList.end(); it++)
	{
		featureMap_E[it->getName()] = (int **)malloc(sizeof(int *) * Num);

		for (int i = 0; i < Num; i++)
		{
			featureMap_E[it->getName()][i] = (int *)malloc(sizeof(int) * Num);
		}

		for (int i = 0; i < Num; i++)
		{
			for (int j = 0; j < Num; j++)
			{
				featureMap_E[it->getName()][i][j] = 0;
			}
		}
	}
}