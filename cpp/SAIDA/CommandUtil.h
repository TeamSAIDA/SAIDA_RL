/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once

#include "Common.h"
#include "Config.h"

namespace MyBot
{
	struct Rect
	{
		int x, y;
		int height, width;
	};

	/// 이동 (move), 공격 (attack), 수리 (repair), 우클릭 (rightClick)  등 유닛 컨트롤 명령을 내릴 때 각종 체크해야할 사항들을 체크한 후 명령 내리도록 하는 헬퍼 함수들
	namespace CommandUtil
	{
		/// attacker 가 target 을 공격하도록 명령 합니다
		bool attackUnit(BWAPI::Unit attacker, BWAPI::Unit target, bool repeat = false);

		/// attacker 가 targetPosition 을 향해 공격 가도록 명령 합니다
		bool attackMove(BWAPI::Unit attacker, const BWAPI::Position &targetPosition, bool repeat = false);

		/// attacker 가 targetPosition 을 향해 이동 가도록 명령 합니다
		void move(BWAPI::Unit attacker, const BWAPI::Position &targetPosition, bool repeat = false);

		/// unit 이 target 에 대해 어떤 행위를 하도록 명령 합니다<br>
		/// 일꾼 유닛이 Mineral Field 에게 : Mineral 자원 채취<br>
		/// 일꾼 유닛이 Refinery 건물에게 : Gas 자원 채취<br>
		/// 전투 유닛이 다른 아군 유닛에게 : Move 명령<br>
		/// 전투 유닛이 다른 적군 유닛에게 : Attack 명령<br>
		void rightClick(BWAPI::Unit unit, BWAPI::Unit target, bool repeat = false, bool rightClickOnly = false);
		void rightClick(BWAPI::Unit unit, BWAPI::Position target, bool repeat = false);

		/// unit 이 target 에 대해 수리 하도록 명령 합니다
		void repair(BWAPI::Unit unit, BWAPI::Unit target, bool repeat = false);

		void patrol(BWAPI::Unit patroller, const BWAPI::Position &targetPosition, bool repeat = false);
		void hold(BWAPI::Unit holder, bool repeat = false);
		void holdControll(BWAPI::Unit unit, BWAPI::Unit target, BWAPI::Position targetPosition, bool targetUnit = false);
		bool build(BWAPI::Unit builder, BWAPI::UnitType building, BWAPI::TilePosition buildPosition);
		void gather(BWAPI::Unit worker, BWAPI::Unit target);
	};


	namespace UnitUtil
	{
		bool IsCombatUnit(BWAPI::Unit unit);
		bool IsValidUnit(BWAPI::Unit unit);
		bool CanAttack(BWAPI::Unit attacker, BWAPI::Unit target);
		double CalculateLTD(BWAPI::Unit attacker, BWAPI::Unit target);
		// attacker 가 target 을 공격할때 사거리를 반환한다. (업그레이드 포함, 벙커에 들어갈때 사거리 증가는 반영 안됨.)
		// 주의 Unit 으로 사용하는 경우, 시야에서 사라지면 잘못된 값 반환
		int GetAttackRange(BWAPI::Unit attacker, BWAPI::Unit target);
		int GetAttackRange(BWAPI::Unit attacker, bool isTargetFlying);
		int GetAttackRange(BWAPI::UnitType attackerType, BWAPI::Player attackerPlayer, bool isFlying);

		// attacker 가 target 을 공격할때 사용하는 weaponType 을 반환한다.
		// 주의 Unit 으로 사용하는 경우, 시야에서 사라지면 잘못된 값 반환
		BWAPI::WeaponType GetWeapon(BWAPI::Unit attacker, BWAPI::Unit target);
		BWAPI::WeaponType GetWeapon(BWAPI::Unit attacker, bool isTargetFlying);
		BWAPI::WeaponType GetWeapon(BWAPI::UnitType attackerType, bool isFlying);

		size_t GetAllUnitCount(BWAPI::UnitType type);

		BWAPI::Unit GetClosestUnitTypeToTarget(BWAPI::UnitType type, BWAPI::Position target);
		double GetDistanceBetweenTwoRectangles(Rect &rect1, Rect &rect2);

		BWAPI::Position GetAveragePosition(std::vector<BWAPI::Unit>  units);
		BWAPI::Unit GetClosestEnemyTargetingMe(BWAPI::Unit myUnit, std::vector<BWAPI::Unit>  units);

		BWAPI::Position getPatrolPosition(BWAPI::Unit attackUnit, BWAPI::WeaponType weaponType, BWAPI::Position targetPos);
	};
}