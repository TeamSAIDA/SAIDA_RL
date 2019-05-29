/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once

#include "Common.h"
#include "InformationManager.h"

#define NOT_DANGER 1000

namespace MyBot
{
	vector<Position> getWidePositions(Position source, Position target, bool forward = true, int gap = TILE_SIZE, int angle = 30, int cnt = 5);
	vector<Position> getRoundPositions(Position source, int gap = TILE_SIZE, int angle = 30);
	Position getDirectionDistancePosition(Position source, Position direction, int distance = TILE_SIZE);

	// Back Position 관련 API
	Position getBackPostion(UnitInfo *unit, Position target, int length, bool avoidUnit = false);
	void moveBackPostion(UnitInfo *unit, Position ePos, int length);
	bool isValidPath(Position st, Position en);
	int getPathValue(Position st, Position en);
	int getPathValueForAir(Position en);
	int getGroundDistance(Position st, Position en);
	int getAltitude(Position pos);
	int getAltitude(TilePosition pos);
	int getAltitude(WalkPosition pos);

	// 공중 유닛 상대 포톤 등 회피 이동 코드
	//void GoWithoutDamage(Unit unit, Position pos);
	//void makeLine_dh(Unit unit, Unit target, double *m, Position pos);
	//void drawLine_dh(Position unit, double m);

	bool isUseMapSettings();
	void focus(Position pos);
	void restartGame();
	void leaveGame();

	bool isSameArea(UnitInfo *u1, UnitInfo *u2);
	bool isSameArea(const Area *a1, const Area *a2);
	bool isSameArea(TilePosition a1, TilePosition a2);
	bool isSameArea(Position a1, Position a2);

	// 내 유닛이 길을 막고있는지 체크한다.
	bool isBlocked(Unit unit, int size = 32);
	bool isBlocked(const UnitType unitType, Position centerPosition, int size = 32);
	bool isBlocked(const UnitType unitType, TilePosition topLeft, int size = 32);
	bool isBlocked(int top, int left, int bottom, int right, int size = 32);

	// UnitList의 평균 Postion 값
	Position getAvgPosition(uList units);
	// 해당 Position의 +1, -1 Tile 사이의 Random Position
	Position findRandomeSpot(Position p);

	// Unit이 target으로 Tower를 피해가는 Function : direction은 위/아래 임.
	// 가는 곳이 갈수 없는 곳이면 Positions::None을 return함.
	// direction 은 1은 시계방향, -1은 반시계방향
	bool goWithoutDamage(Unit u, Position target, int direction, int dangerGap = 3 * TILE_SIZE);
	void kiting(UnitInfo *attacker, UnitInfo *target, int dangerPoint, int threshold);
	void attackFirstkiting(UnitInfo *attacker, UnitInfo *target, int dangerPoint, int threshold);
	void pControl(UnitInfo *attacker, UnitInfo *target);
	UnitInfo *getGroundWeakTargetInRange(UnitInfo *attacker, bool worker = false);

	// 중심과 각도(도 단위)와 원위의 한 점이 주어지면 원위의 한점에서 각도만큼 떨어진 원위의 또 다른 한 점의 좌표를 구합니다.
	Position getCirclePosFromPosByDegree(Position center, Position fromPos, double degree);
	// 중심과 각도(라디안 단위)와 원위의 한 점이 주어지면 원위의 한점에서 각도만큼 떨어진 원위의 또 다른 한 점의 좌표를 구합니다.
	Position getCirclePosFromPosByRadian(Position center, Position fromPos, double radian);
	// 한점과 거리와 각도(도 단위)가 주어지면 그 점에서 그 각도로 거리만큼 떨어진 점의 좌표를 구합니다.
	Position getPosByPosDistDegree(Position pos, int dist, double degree);
	// 한점과 거리와 각도(라디안 단위)가 주어지면 그 점에서 그 각도로 거리만큼 떨어진 점의 좌표를 구합니다.
	Position getPosByPosDistRadian(Position pos, int dist, double degree);
	// p1 기준으로 p2 의 각도(라디안 단위)를 반환한다.
	double getRadian(Position p1, Position p2);
	double getRadian2(Position p1, Position p2);

	int getDamage(Unit attacker, Unit target);
	int getDamage(UnitType attackerType, UnitType targetType, Player attackerPlayer, Player targetPlayer);
	UnitInfo *getDangerUnitNPoint(Position p, int *point, bool isFlyer);

	// AttackRange 계산 기준으로 거리를 가져온다. weaponRange 와 비교 가능
	int getAttackDistance(int aLeft, int aTop, int aRight, int aBottom, int tLeft, int tTop, int tRight, int tBottom);
	int getAttackDistance(Unit attacker, Unit target);
	int getAttackDistance(Unit attacker, UnitType targetType, Position targetPosition);
	int getAttackDistance(UnitType attackerType, Position attackerPosition, Unit target);
	int getAttackDistance(UnitType attackerType, Position attackerPosition, UnitType targetType, Position targetPosition);

	// 가장 가까운 장애물까지의 거리
	vector<int> getNearObstacle(UnitInfo *uInfo, int directCnt, bool resource = false);
	vector<pair<double, double>> getRadianAndDistanceFromEnemy(UnitInfo *uInfo, int directCnt);
	vector<int> getEnemiesInAngle(UnitInfo *uInfo, uList enemies, int directCnt, int range);
}