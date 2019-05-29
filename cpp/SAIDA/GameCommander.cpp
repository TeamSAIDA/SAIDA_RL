/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "GameCommander.h"
#include "DeepLearning\GymFactory.h"

using namespace MyBot;

GameCommander::GameCommander() {
}
GameCommander::~GameCommander() {
}

GameCommander &GameCommander::Instance()
{
	static GameCommander instance;
	return instance;
}

void GameCommander::onStart()
{
	TilePosition startLocation = Broodwar->self()->getStartLocation();

	if (startLocation == TilePositions::None || startLocation == TilePositions::Unknown) {
		return;
	}

	// singleton initializing 필요한 경우
	InformationManager::Instance().initialize();

#if GYM
	BWML::GymFactory::Instance().InitializeGym();
#endif
}

void GameCommander::onEnd(bool isWinner)
{
}

void GameCommander::update()
{
	if (Broodwar->self() == nullptr || Broodwar->self()->isDefeated() || Broodwar->self()->leftGame()
			|| Broodwar->enemy() == nullptr || Broodwar->enemy()->isDefeated() || Broodwar->enemy()->leftGame()) {
		return;
	}
	else if (Broodwar->isPaused()) {
		UXManager::Instance().update();
		return;
	}

	// 아군 베이스 위치. 적군 베이스 위치. 각 유닛들의 상태정보 등을 Map 자료구조에 저장/업데이트
	InformationManager::Instance().update();
#if GYM
	BWML::GymFactory::Instance().GetGym()->update();
#endif
	UXManager::Instance().update();
}

// BasicBot 1.1 Patch Start ////////////////////////////////////////////////
// 일꾼 탄생/파괴 등에 대한 업데이트 로직 버그 수정 : onUnitShow 가 아니라 onUnitComplete 에서 처리하도록 수정
void GameCommander::onUnitShow(Unit unit)
{
	if (unit->getPlayer() == E && !unit->isCompleted())
		InformationManager::Instance().onUnitShow(unit);
}

// BasicBot 1.1 Patch End //////////////////////////////////////////////////

void GameCommander::onUnitHide(Unit unit)
{
}

void GameCommander::onUnitCreate(Unit unit)
{
	if (unit->getPlayer() == S) {
		InformationManager::Instance().onUnitCreate(unit);
#if GYM
		GymFactory::Instance().GetGym()->onUnitCreate(unit);
#endif
	}
}

// BasicBot 1.1 Patch Start ////////////////////////////////////////////////
// 일꾼 탄생/파괴 등에 대한 업데이트 로직 버그 수정 : onUnitShow 가 아니라 onUnitComplete 에서 처리하도록 수정
void GameCommander::onUnitComplete(Unit unit)
{
	InformationManager::Instance().onUnitComplete(unit);
#if GYM
	GymFactory::Instance().GetGym()->onUnitComplete(unit);
#endif
}

// BasicBot 1.1 Patch End //////////////////////////////////////////////////

void GameCommander::onUnitDestroy(Unit unit)
{
#if GYM
	GymFactory::Instance().GetGym()->onUnitDestroy(unit);
#endif
	InformationManager::Instance().onUnitDestroy(unit);
}

void GameCommander::onUnitRenegade(Unit unit)
{

}

void GameCommander::onUnitMorph(Unit unit)
{
}

void GameCommander::onUnitDiscover(Unit unit)
{
}

void GameCommander::onUnitEvade(Unit unit)
{
}

void GameCommander::onUnitLifted(Unit unit)
{
}

void GameCommander::onUnitLanded(Unit unit)
{
}

// BasicBot 1.1 Patch Start ////////////////////////////////////////////////
// onNukeDetect, onPlayerLeft, onSaveGame 이벤트를 처리할 수 있도록 메소드 추가

void GameCommander::onNukeDetect(Position target)
{
}

void GameCommander::onPlayerLeft(Player player)
{
}

void GameCommander::onSaveGame(string gameName)
{
}

// BasicBot 1.1 Patch End //////////////////////////////////////////////////

void GameCommander::onSendText(string text)
{
}

void GameCommander::onReceiveText(Player player, string text)
{
}
