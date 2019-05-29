/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once

#include "Common.h"
#include "UnitData/UnitData.h"
#include "InformationManager.h"
#include "AbstractManager.h"

namespace MyBot
{
	/// 봇 프로그램 개발의 편의성 향상을 위해 게임 화면에 추가 정보들을 표시하는 class<br>
	/// 여러 Manager 들로부터 정보를 조회하여 Screen 혹은 Map 에 정보를 표시합니다
	class UXManager : public AbstractManager
	{
	private:
		UXManager();

		const int dotRadius = 2;

		// 게임 개요 정보를 Screen 에 표시합니다
		void drawGameInformationOnScreen(int x, int y);

		/// APM (Action Per Minute) 숫자를 Screen 에 표시합니다
		void drawAPM(int x, int y);

		/// Players 정보를 Screen 에 표시합니다
		void drawPlayers();

		/// Player 들의 팀 (Force) 들의 정보를 Screen 에 표시합니다
		void drawForces();

		/// Build 진행 상태를 Screen 에 표시합니다
		void drawBuildStatusOnScreen(int x, int y);


		/// UnitType 별 통계 정보를 Screen 에 표시합니다
		//		void drawUnitStatisticsOnScreen(int x, int y);
		//		UnitData에 추가후 수정 필요. gangoku 02.05

		/// Unit 의 Id 를 Map 에 표시합니다
		void drawUnitIdOnMap();

		/// Unit 의 Target 으로 잇는 선을 Map 에 표시합니다
		void drawUnitTargetOnMap();

		/// Bullet 을 Map 에 표시합니다
		/// Cloaking Unit 의 Bullet 표시에 쓰입니다
		void drawBulletsOnMap();


		/// UnitData 정보를 Screen 에 표시합니다
		void drawAllUnitData(int x, int y);


		void drawCoolDown(std::pair<const BWAPI::Unit, MyBot::UnitInfo *> &u);

	protected:
		void updateManager() override;

	public:
		/// static singleton 객체를 리턴합니다
		static UXManager 	&Instance();

		/// 경기가 시작될 때 일회적으로 추가 정보를 출력합니다
		void onStart();
		void drawUnitHP(UnitInfo *u);
	};

	/// 빌드 진행상황을 빌드 시작 순서대로 정렬하여 표시하기 위한 Comparator class
	class CompareWhenStarted
	{
	public:

		CompareWhenStarted() {}

		/// 빌드 진행상황을 빌드 시작 순서대로 정렬하여 표시하기 위한 sorting operator
		bool operator() (Unit u1, Unit u2)
		{
			int startedU1 = Broodwar->getFrameCount() - (u1->getType().buildTime() - u1->getRemainingBuildTime());
			int startedU2 = Broodwar->getFrameCount() - (u2->getType().buildTime() - u2->getRemainingBuildTime());
			return startedU1 > startedU2;
		}
	};

}
