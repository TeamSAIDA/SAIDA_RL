/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once

#include "Common.h"

#include "AbstractManager.h"
#include "UnitData/UnitData.h"
#include "SaidaUtil.h"

#define INFO	InformationManager::Instance()
#define MYBASE INFO.getMainBaseLocation(S)->Center()
#define ENBASE INFO.getMainBaseLocation(E)->Center()

namespace MyBot
{
	enum TypeKind
	{
		AllUnitKind,
		AirUnitKind,
		GroundCombatKind, //나중에..
		GroundUnitKind,
		BuildingKind,
		AllDefenseBuildingKind,
		AirDefenseBuildingKind,
		GroundDefenseBuildingKind,
		AllKind
	};

	/// 게임 상황정보 중 일부를 자체 자료구조 및 변수들에 저장하고 업데이트하는 class<br>
	/// 현재 게임 상황정보는 Broodwar 를 조회하여 파악할 수 있지만, 과거 게임 상황정보는 Broodwar 를 통해 조회가 불가능하기 때문에 InformationManager에서 별도 관리하도록 합니다<br>
	/// 또한, Broodwar 나 BWEM 등을 통해 조회할 수 있는 정보이지만 전처리 / 별도 관리하는 것이 유용한 것도 InformationManager에서 별도 관리하도록 합니다
	class InformationManager : public AbstractManager
	{
		InformationManager() : AbstractManager("InformationManager") {};
		~InformationManager() {};

		/// 맵 플레이어수 (2인용 맵, 3인용 맵, 4인용 맵, 8인용 맵)
		int															mapPlayerLimit;

		/// Player - UnitData(각 Unit 과 그 Unit의 UnitInfo 를 Map 형태로 저장하는 자료구조) 를 저장하는 자료구조 객체<br>
		map<Player, UnitData>							_unitData;

		/// 전체 unit 의 정보를 업데이트 합니다 (UnitType, lastPosition, HitPoint 등. 프레임당 1회 실행)
		void                    updateUnitsInfo();
		// enemy 전용 ( Hide, Show )

		set<TechType>			researchedSet;
		map<UpgradeType, int>		upgradeSet;
		vector<UpgradeType>		upgradeList;

	protected:
		void updateManager();

	public:

		/// static singleton 객체를 리턴합니다
		static InformationManager &Instance();
		void initialize();

		Player       selfPlayer;		///< 아군 Player
		Race			selfRace;		///< 아군 Player의 종족
		Player       enemyPlayer;	///< 적군 Player
		Race			enemyRace;		///< 적군 Player의 종족
		Race		enemySelectRace;		///< 적군 Player가 선택한 종족

		/// Unit 에 대한 정보를 업데이트합니다
		void					onUnitShow(Unit unit);
		/// Unit 에 대한 정보를 업데이트합니다
		//		void					onUnitHide(Unit unit)        { updateUnitHide(unit, true); }
		/// Unit 에 대한 정보를 업데이트합니다
		void					onUnitCreate(Unit unit);
		/// Unit 에 대한 정보를 업데이트합니다
		void					onUnitComplete(Unit unit);
		/// 유닛이 파괴/사망한 경우, 해당 유닛 정보를 삭제합니다
		void					onUnitDestroy(Unit unit);

		/// 현재 맵의 최대 플레이어수 (2인용 맵, 3인용 맵, 4인용 맵, 8인용 맵) 을 리턴합니다
		int						getMapPlayerLimit() {
			return mapPlayerLimit;
		}

		/// 해당 Player (아군 or 적군) 의 모든 유닛 통계 UnitData 을 리턴합니다
		UnitData 				&getUnitData(Player player) {
			return _unitData[player];
		}

		// 해당 종족의 UnitType 중 ResourceDepot 기능을 하는 UnitType을 리턴합니다
		UnitType			getBasicResourceDepotBuildingType(Race race = Races::None);

		// 해당 종족의 UnitType 중 Refinery 기능을 하는 UnitType을 리턴합니다
		UnitType			getRefineryBuildingType(Race race = Races::None);

		// 해당 종족의 UnitType 중 SupplyProvider 기능을 하는 UnitType을 리턴합니다
		UnitType			getBasicSupplyProviderUnitType(Race race = Races::None);

		// 해당 종족의 UnitType 중 Worker 에 해당하는 UnitType을 리턴합니다
		UnitType			getWorkerType(Race race = Races::None);

		// 해당 종족의 UnitType 중 Basic Combat Unit 에 해당하는 UnitType을 리턴합니다
		UnitType			getBasicCombatUnitType(Race race = Races::None);

		// 해당 종족의 UnitType 중 Basic Combat Unit 을 생산하기 위해 건설해야하는 UnitType을 리턴합니다
		UnitType			getBasicCombatBuildingType(Race race = Races::None);

		// 해당 종족의 UnitType 중 Advanced Combat Unit 에 해당하는 UnitType을 리턴합니다
		UnitType			getAdvancedCombatUnitType(Race race = Races::None);

		// 해당 종족의 UnitType 중 Observer 에 해당하는 UnitType을 리턴합니다
		UnitType			getObserverUnitType(Race race = Races::None);

		// 해당 종족의 UnitType 중 Advanced Depense 기능을 하는 UnitType을 리턴합니다
		UnitType			getAdvancedDefenseBuildingType(Race race = Races::None, bool isAirDefense = false);


		// UnitData 관련 API는 아래에서 정의한다.
		UnitInfo *getUnitInfo(Unit unit, Player p);
		uList getUnits(UnitType t, Player p) {
			return _unitData[p].getUnitVector(t);
		}
		uList getBuildings(UnitType t, Player p) {
			return _unitData[p].getBuildingVector(t);
		}
		uMap &getUnits(Player p) {
			return _unitData[p].getAllUnits();
		}
		uMap &getBuildings(Player p) {
			return _unitData[p].getAllBuildings();
		}
		int			getCompletedCount(UnitType t, Player p);
		int			getDestroyedCount(UnitType t, Player p);
		int			getTotalCount(UnitType t, Player p) {
			return getAllCount(t, p) + getDestroyedCount(t, p);
		}
		map<UnitType, int> getDestroyedCountMap(Player p);
		int			getAllCount(UnitType t, Player p);

		void clearUnitNBuilding();

		uList getUnitsInRadius(Player p, Position pos = Positions::Origin, int radius = 0, bool ground = true, bool air = true, bool worker = true, bool hide = false, bool groundDist = false);
		uList getBuildingsInRadius(Player p, Position pos = Positions::Origin, int radius = 0, bool ground = true, bool air = true, bool hide = false, bool groundDist = false);
		uList getAllInRadius(Player p, Position pos = Positions::Origin, int radius = 0, bool ground = true, bool air = true, bool hide = false, bool groundDist = false);
		uList getUnitsInRectangle(Player p, Position leftTop, Position rightDown, bool ground = true, bool air = true, bool worker = true, bool hide = false);
		uList getBuildingsInRectangle(Player p, Position leftTop, Position rightDown, bool ground = true, bool air = true, bool hide = false);
		uList getAllInRectangle(Player p, Position leftTop, Position rightDown, bool ground = true, bool air = true, bool hide = false);
		uList getTypeUnitsInRadius(UnitType t, Player p, Position pos = Positions::Origin, int radius = 0, bool hide = false);
		uList getTypeBuildingsInRadius(UnitType t, Player p, Position pos = Positions::Origin, int radius = 0, bool incomplete = true, bool hide = true);
		uList getDefenceBuildingsInRadius(Player p, Position pos = Positions::Origin, int radius = 0, bool incomplete = true, bool hide = true);
		uList getTypeUnitsInRectangle(UnitType t, Player p, Position leftTop, Position rightDown, bool hide = false);
		uList getTypeBuildingsInRectangle(UnitType t, Player p, Position leftTop, Position rightDown, bool incomplete = true, bool hide = true);
		uList getUnitsInArea(Player p, Position pos, bool ground = true, bool air = true, bool worker = true, bool hide = true);
		uList getBuildingsInArea(Player p, Position pos, bool ground = true, bool air = true, bool hide = true);
		uList getTypeUnitsInArea(UnitType t, Player p, Position pos, bool hide = false);
		uList getTypeBuildingsInArea(UnitType t, Player p, Position pos, bool incomplete = true, bool hide = true);

		UnitInfo *getClosestUnit(Player p, Position pos, TypeKind kind = TypeKind::AllKind, int radius = 0, bool worker = false, bool hide = false, bool groundDist = false, bool detectedOnly = true);
		UnitInfo *getFarthestUnit(Player p, Position pos, TypeKind kind = TypeKind::AllKind, int radius = 0, bool worker = false, bool hide = false, bool groundDist = false, bool detectedOnly = true);
		UnitInfo *getClosestTypeUnit(Player p, Position pos, UnitType type, int radius = 0, bool hide = false, bool groundDist = false, bool detectedOnly = true);
		UnitInfo *getClosestTypeUnit(Player p, Position pos, vector<UnitType> &types, int radius = 0, bool hide = false, bool groundDist = false, bool detectedOnly = true);
		UnitInfo *getFarthestTypeUnit(Player p, Position pos, UnitType type, int radius = 0, bool hide = false, bool groundDist = false, bool detectedOnly = true);

		bool hasResearched(TechType tech);
		void setResearched(UnitType unitType);
		void setUpgradeLevel();
		int getUpgradeLevel(UpgradeType up);

		vector<UnitType> getCombatTypes(Race race);
	};
}
