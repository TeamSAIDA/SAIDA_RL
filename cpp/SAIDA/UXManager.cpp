/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "UXManager.h"

using namespace MyBot;

UXManager::UXManager() : AbstractManager("UXManager")
{

}

UXManager &UXManager::Instance()
{
	static UXManager instance;
	return instance;
}


void UXManager::onStart()
{
}

void UXManager::updateManager() {

	// 공격
	if (Config::Debug::DrawUnitTargetInfo)
	{
		drawUnitTargetOnMap();

		// 미사일, 럴커의 보이지않는 공격등을 표시
		drawBulletsOnMap();
	}

	//cout << 14;

	// draw tile position of mouse cursor
	if (Config::Debug::DrawMouseCursorInfo)
	{
		int mouseX = Broodwar->getMousePosition().x + Broodwar->getScreenPosition().x;
		int mouseY = Broodwar->getMousePosition().y + Broodwar->getScreenPosition().y;

		int mouseDrawX = mouseX;
		int mouseDrawY = mouseY;

		if (theMap.Size().x - mouseDrawX / 32 < 5)
			mouseDrawX -= 100;

		if (theMap.Size().y - mouseDrawY / 32 < 6)
			mouseDrawY -= 40;

		Broodwar->drawTextMap(mouseDrawX + 20, mouseDrawY, "(%d, %d)", (int)(mouseX / TILE_SIZE), (int)(mouseY / TILE_SIZE));
	}

	if (Config::Debug::DrawMyUnit || Config::Debug::DrawEnemyUnit)
	{
		drawAllUnitData(20, 20);
	}

	drawUnitIdOnMap();
}

void UXManager::drawAllUnitData(int x, int y)
{
	Broodwar->drawTextScreen(x, y, "\x04 <UnitData>");

	map<UnitType, uList> &allSUnit = INFO.getUnitData(S).getUnitTypeMap();
	map<UnitType, uList> &allSBuild = INFO.getUnitData(S).getBuildingTypeMap();
	uMap &sUnit = INFO.getUnits(S);
	uMap &sBuild = INFO.getBuildings(S);

	map<UnitType, uList> &allEUnit = INFO.getUnitData(E).getUnitTypeMap();
	map<UnitType, uList> &allEBuild = INFO.getUnitData(E).getBuildingTypeMap();
	uMap &eUnit = INFO.getUnits(E);
	uMap &eBuild = INFO.getBuildings(E);

	map<UnitType, uList>::iterator iter;

	if (Config::Debug::DrawMyUnit)
	{
		int gap = 10;

		for (iter = allSUnit.begin(); iter != allSUnit.end(); iter++)
		{
			if (iter->second.empty())
				continue;

			Broodwar->drawTextScreen(x, y + gap, "%s : %d", iter->first.getName().c_str(), iter->second.size());
			gap += 10;
		}

		Broodwar->drawTextScreen(x, y + gap, "AllUnit : %d", sUnit.size());
		gap += 20;

		for (iter = allSBuild.begin(); iter != allSBuild.end(); iter++)
		{
			if (iter->second.empty())
				continue;

			Broodwar->drawTextScreen(x, y + gap, "%s : %d", iter->first.getName().c_str(), iter->second.size());
			gap += 10;
		}

		Broodwar->drawTextScreen(x, y + gap, "AllBuildings : %d", sBuild.size());

		x += 180;

		gap = 0;
		Broodwar->drawTextScreen(x, y + gap, "\x04 <Unit Create/All>");
		gap += 10;

		for (iter = allSUnit.begin(); iter != allSUnit.end(); iter++)
		{
			if (INFO.getCompletedCount(iter->first, S) + INFO.getAllCount(iter->first, S) == 0)
				continue;

			Broodwar->drawTextScreen(x, y + gap, "%s : %d/%d", iter->first.getName().c_str(), INFO.getCompletedCount(iter->first, S), INFO.getAllCount(iter->first, S));
			gap += 10;
		}

		gap += 10;
		Broodwar->drawTextScreen(x, y + gap, "\x04 <Building Create/All>");
		gap += 10;

		for (iter = allSBuild.begin(); iter != allSBuild.end(); iter++)
		{
			if (INFO.getCompletedCount(iter->first, S) + INFO.getAllCount(iter->first, S) == 0)
				continue;

			Broodwar->drawTextScreen(x, y + gap, "%s : %d/%d", iter->first.getName().c_str(), INFO.getCompletedCount(iter->first, S), INFO.getAllCount(iter->first, S));
			gap += 10;
		}
	}
	else
	{
		int gap = 10;

		for (iter = allEUnit.begin(); iter != allEUnit.end(); iter++)
		{
			if (iter->second.empty())
				continue;

			Broodwar->drawTextScreen(x, y + gap, "%s : %d", iter->first.getName().c_str(), iter->second.size());
			gap += 10;
		}

		Broodwar->drawTextScreen(x, y + gap, "AllUnit : %d", eUnit.size());
		gap += 20;

		for (iter = allEBuild.begin(); iter != allEBuild.end(); iter++)
		{
			if (iter->second.empty())
				continue;

			Broodwar->drawTextScreen(x, y + gap, "%s : %d", iter->first.getName().c_str(), iter->second.size());
			gap += 10;
		}

		Broodwar->drawTextScreen(x, y + gap, "AllBuildings : %d", eBuild.size());

		x += 180;

		gap = 0;
		Broodwar->drawTextScreen(x, y + gap, "\x04 <Unit Complete/All>");
		gap += 10;

		for (iter = allEUnit.begin(); iter != allEUnit.end(); iter++)
		{
			if (INFO.getCompletedCount(iter->first, E) + INFO.getAllCount(iter->first, E) == 0)
				continue;

			Broodwar->drawTextScreen(x, y + gap, "%s : %d/%d", iter->first.getName().c_str(), INFO.getCompletedCount(iter->first, E), INFO.getAllCount(iter->first, E) );
			gap += 10;
		}

		gap += 10;
		Broodwar->drawTextScreen(x, y + gap, "\x04 <Building Complete/All>");
		gap += 10;

		for (iter = allEBuild.begin(); iter != allEBuild.end(); iter++)
		{
			if (INFO.getCompletedCount(iter->first, E) + INFO.getAllCount(iter->first, E) == 0)
				continue;

			Broodwar->drawTextScreen(x, y + gap, "%s : %d/%d", iter->first.getName().c_str(), INFO.getCompletedCount(iter->first, E), INFO.getAllCount(iter->first, E));
			gap += 10;
		}
	}
}

void UXManager::drawGameInformationOnScreen(int x, int y)
{
	Broodwar->drawTextScreen(x, y, "\x04Players:");
	Broodwar->drawTextScreen(x + 50, y, "%c%s(%s) \x04vs. %c%s(%s)",
							 S->getTextColor(), S->getName().c_str(), INFO.selfRace.c_str(),
							 E->getTextColor(), E->getName().c_str(), INFO.enemyRace.c_str());
	y += 12;

	Broodwar->drawTextScreen(x, y, "\x04Map:");
	Broodwar->drawTextScreen(x + 50, y, "\x03%s (%d x %d size)", Broodwar->mapFileName().c_str(), Broodwar->mapWidth(), Broodwar->mapHeight());
	Broodwar->setTextSize();
	y += 12;

	Broodwar->drawTextScreen(x, y, "\x04Time:");
	Broodwar->drawTextScreen(x + 50, y, "\x04%d", Broodwar->getFrameCount());
	Broodwar->drawTextScreen(x + 90, y, "\x04%4d:%3d", (int)(Broodwar->getFrameCount() / (23.8 * 60)), (int)((int)(Broodwar->getFrameCount() / 23.8) % 60));
}

void UXManager::drawAPM(int x, int y)
{
	int bwapiAPM = Broodwar->getAPM();
	Broodwar->drawTextScreen(x, y, "APM : %d", bwapiAPM);
}

void UXManager::drawPlayers()
{
	Playerset players = Broodwar->getPlayers();

	for (auto p : players)
		Broodwar << "Player [" << p->getID() << "]: " << p->getName() << " is in force: " << p->getForce()->getName() << endl;
}

void UXManager::drawForces()
{
	Forceset forces = Broodwar->getForces();

	for (auto f : forces)
	{
		Playerset players = f->getPlayers();
		Broodwar << "Force " << f->getName() << " has the following players:" << endl;

		for (auto p : players)
			Broodwar << "  - Player [" << p->getID() << "]: " << p->getName() << endl;
	}
}

void UXManager::drawBuildStatusOnScreen(int x, int y)
{
	// 건설 / 훈련 중인 유닛 진행상황 표시
	vector<Unit> unitsUnderConstruction;

	for (auto &unit : S->getUnits())
	{
		if (unit != nullptr && unit->isBeingConstructed())
		{
			unitsUnderConstruction.push_back(unit);
		}
	}

	// sort it based on the time it was started
	sort(unitsUnderConstruction.begin(), unitsUnderConstruction.end(), CompareWhenStarted());

	Broodwar->drawTextScreen(x, y, "\x04 <Build Status>");

	size_t reps = unitsUnderConstruction.size() < 10 ? unitsUnderConstruction.size() : 10;

	string prefix = "\x07";

	for (auto &unit : unitsUnderConstruction)
	{
		y += 10;
		UnitType t = unit->getType();

		if (t == UnitTypes::Zerg_Egg)
		{
			t = unit->getBuildType();
		}

		Broodwar->drawTextScreen(x, y, " %s%s (%d)", prefix.c_str(), t.getName().c_str(), unit->getRemainingBuildTime());
	}

	// Tech Research 표시

	// Upgrade 표시

}

void UXManager::drawUnitIdOnMap() {
	for (auto &unit : S->getUnits())
	{
		if (Special_Map_Revealer != unit->getType())
			Broodwar->drawTextMap(unit->getPosition().x, unit->getPosition().y - 12, "\x07%d", unit->getID());
	}

	for (auto &unit : E->getUnits())
	{

		if (!unit->isDetected() && unit->getType()) {
			bw->drawBoxMap(unit->getLeft(), unit->getTop(), unit->getRight(), unit->getBottom(), Colors::Red, false);
			Broodwar->drawTextMap(unit->getPosition().x, unit->getPosition().y - 12, "\x06%d %s", unit->getID(), unit->getType().c_str());
		}
		else {
			Broodwar->drawTextMap(unit->getPosition().x, unit->getPosition().y - 12, "\x06%d", unit->getID());
		}

	}

	for (auto &unit : BWAPI::Broodwar->neutral()->getUnits())
	{
		BWAPI::Broodwar->drawTextMap(unit->getPosition().x, unit->getPosition().y - 12, "\x04%d", unit->getID());
	}
}

void UXManager::drawCoolDown(std::pair<const BWAPI::Unit, MyBot::UnitInfo *> &u)
{
	int maxCoolDown = u.first->getType().airWeapon().targetsAir() ? u.first->getType().airWeapon().damageCooldown() : u.first->getType().groundWeapon().damageCooldown();
	int currentCoolDown = u.first->getType().airWeapon().targetsAir() ? u.first->getAirWeaponCooldown() : u.first->getGroundWeaponCooldown();

	if (maxCoolDown != 0) {
		// 쿨다운 표시
		int barSizeX = u.second->type().width();
		int barSizeY = 5;
		int positionY = u.second->pos().y + 12;
		int positionX = u.second->pos().x - barSizeX / 2;

		double cooldown = (double)(maxCoolDown - currentCoolDown) / maxCoolDown;
		Color color = cooldown == 1 ? Color(20, 200, 50) : Color(200, 20, 20);
		bw->drawBoxMap(Position(positionX, positionY), Position(positionX + barSizeX + 2, positionY + barSizeY), Colors::Black, true);
		bw->drawBoxMap(Position(positionX + 1, positionY + 1), Position(positionX + 1 + (int)(barSizeX * cooldown), positionY + barSizeY - 1), color, true);
	}
}

void UXManager::drawUnitTargetOnMap()
{
	for (auto &unit : S->getUnits())
	{
		if (unit != nullptr && unit->isCompleted() && !unit->getType().isBuilding() && !unit->getType().isWorker())
		{
			Unit targetUnit = unit->getTarget();

			if (targetUnit != nullptr && targetUnit->getPlayer() != S) {
				Broodwar->drawCircleMap(unit->getPosition(), dotRadius, Colors::Red, true);
				Broodwar->drawCircleMap(targetUnit->getTargetPosition(), dotRadius, Colors::Red, true);
				Broodwar->drawLineMap(unit->getPosition(), targetUnit->getTargetPosition(), Colors::Red);
			}
			else if (unit->isMoving()) {
				Broodwar->drawCircleMap(unit->getPosition(), dotRadius, Colors::Orange, true);
				Broodwar->drawCircleMap(unit->getTargetPosition(), dotRadius, Colors::Orange, true);
				Broodwar->drawLineMap(unit->getPosition(), unit->getTargetPosition(), Colors::Orange);
			}

		}
	}
}

// Bullet 을 Line 과 Text 로 표시한다. Cloaking Unit 의 Bullet 표시에 쓰인다
void UXManager::drawBulletsOnMap()
{
	for (auto &b : Broodwar->getBullets())
	{
		Position p = b->getPosition();
		double velocityX = b->getVelocityX();
		double velocityY = b->getVelocityY();

		// 아군 것이면 녹색, 적군 것이면 빨간색
		Broodwar->drawLineMap(p, p + Position((int)velocityX, (int)velocityY), b->getPlayer() == S ? Colors::Green : Colors::Red);

		Broodwar->drawTextMap(p, "%c%s", b->getPlayer() == S ? Text::Green : Text::Red, b->getType().c_str());
	}
}

void UXManager::drawUnitHP(UnitInfo *u)
{
	int maxHP = u->type().maxHitPoints() + u->type().maxShields();
	int currentHP = u->hp();

	// 체력 표시
	int barSizeX = u->type().width();
	int barSizeY = 5;
	int positionY = u->pos().y + 12;
	int positionX = u->pos().x - barSizeX / 2;

	double hp = (double)currentHP / maxHP;
	Color color = hp > 0.5 ? Color(20, 200, 50) : Color(200, 20, 20);
	bw->drawBoxMap(Position(positionX, positionY), Position(positionX + barSizeX + 2, positionY + barSizeY), Colors::Black, true);
	bw->drawBoxMap(Position(positionX + 1, positionY + 1), Position(positionX + 1 + (int)(barSizeX * hp), positionY + barSizeY - 1), color, true);
}