/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */


#include "UnitInfo.h"
#include "../GameCommander.h"
#include "../InformationManager.h"

using namespace MyBot;

UnitInfo::UnitInfo(Unit unit)
{
	m_unit = unit;

	m_type = m_unit->getType();
	m_unitID = m_unit->getID();
	m_player = m_unit->getPlayer();
	m_lastPositionTime = TIME;
	m_lastPosition = m_unit->getPosition();
	m_lastSeenPosition = m_lastPosition;
	m_completed = m_unit->isCompleted();
	m_hide = true;
	m_lifted = false;
	m_vPosition = Positions::None;
	m_avgEnemyPosition = Positions::None;
	m_veryFrontEnemUnit = nullptr;
	m_beingRepaired = false;
	m_LastCommandFrame = 0;
	m_morphing = m_unit->isMorphing();
	m_burrowed = m_unit->isBurrowed();
	m_canBurrow = m_unit->canBurrow(false);
	m_isPowered = true;
	m_blockedCnt = 0;
	m_cooldown = 0;

	if (m_unit->isCompleted()) {
		m_HP = m_type.maxHitPoints();
		m_Shield = m_type.maxShields();
	}
	else {
		m_HP = 0;
		m_Shield = 0;
	}

	m_expectedDamage = 0;

	if (m_unit->getPlayer() == S) {
		m_spaceRemaining = m_unit->getSpaceRemaining();
		m_energy = m_unit->getEnergy();
	}
	else {
		m_spaceRemaining = 0;
		m_energy = m_type.maxEnergy();
	}

	m_lastNearUnitFrame = 0;
	m_nearUnitFrameCnt = 0;
}

void UnitInfo::Update()
{
	if (m_unit->exists() && !m_unit->isLoaded())
	{
		bool isShowThisFrame = m_hide;

		m_hide = false;

		// 공격을 하고 있을때는 블락킹 처리를 하지 않는다.
		if (m_lastPosition == m_unit->getPosition() && m_vPosition != m_lastPosition && m_unit->getGroundWeaponCooldown() == 0) {
			m_blockedQueue.push(true);
			m_blockedCnt++;

			if (m_blockedQueue.size() > 120) {
				if (m_blockedQueue.front())
					m_blockedCnt--;

				m_blockedQueue.pop();
			}
		}
		else {
			m_blockedQueue.push(false);

			if (m_blockedQueue.size() > 120) {
				if (m_blockedQueue.front())
					m_blockedCnt--;

				m_blockedQueue.pop();
			}
		}

		if (m_lastPosition != m_unit->getPosition()) {
			m_lastPositionTime = TIME;
			m_lastPosition = m_unit->getPosition();
			m_lastSeenPosition = m_lastPosition;
		}
		else if (m_unit->getAirWeaponCooldown() || m_unit->getGroundWeaponCooldown() || m_unit->getSpellCooldown() || m_unit->isConstructing() || m_unit->isHoldingPosition() ) {
			m_lastPositionTime = TIME;
		}

		m_vPosition = m_lastPosition + Position((int)(m_unit->getVelocityX() * 8), (int)(m_unit->getVelocityY() * 8));
		m_completed = m_unit->isCompleted();
		m_morphing = m_unit->isMorphing();

		if (m_unit->getHitPoints() > m_HP)
			m_beingRepaired = true;
		else
			m_beingRepaired = false;

		if (m_unit->isDetected()) {
			m_HP = m_unit->getHitPoints();
			m_Shield = m_unit->getShields();
		}

		m_cooldown = max(m_unit->getGroundWeaponCooldown(), m_unit->getAirWeaponCooldown());
		m_type = m_unit->getType();
		m_expectedDamage = 0;
		m_isPowered = m_unit->isPowered();

		if (m_unit->getPlayer() == S) {
			m_energy = m_unit->getEnergy();
			m_spaceRemaining = m_unit->getSpaceRemaining();
		}
		else {
			m_energy = min((double)E->maxEnergy(m_type), m_energy + 0.03125);
		}

		if ( m_type.isBurrowable())
		{
			if (m_canBurrow == true && m_unit->canBurrow(false) == false)
				m_burrowed = true;
			else if (m_canBurrow == false && m_unit->canBurrow(false) == true)
				m_burrowed = false;

			m_canBurrow = m_unit->canBurrow(false);
		}

		m_lifted = m_unit->isFlying();

		if (m_unit->getPlayer() != S && m_type.canAttack())
		{
			if (m_unit->isSelected()) {
				if (m_unit->getOrderTarget() != nullptr)
					cout << TIME << " Order " << m_unit->getOrderTarget()->getType() << " " << m_unit->getOrderTarget()->getTilePosition() << endl;

				if (m_unit->getTarget() != nullptr)
					cout << TIME << " Target " << m_unit->getTarget()->getType() << " " << m_unit->getTarget()->getTilePosition() << endl;
			}

			if (m_unit->getOrderTarget() != nullptr && m_unit->getOrderTarget()->exists() && m_unit->getOrderTarget()->getPlayer() == S)
			{
				if (INFO.getUnitInfo(m_unit->getOrderTarget(), S) != nullptr)
				{
					INFO.getUnitInfo(m_unit->getOrderTarget(), S)->getEnemiesTargetMe().push_back(m_unit);
				}
			}

			else if (m_unit->getTarget() != nullptr && m_unit->getTarget()->exists() && m_unit->getTarget()->getPlayer() == S)
			{
				if (INFO.getUnitInfo(m_unit->getTarget(), S) != nullptr)
				{
					INFO.getUnitInfo(m_unit->getTarget(), S)->getEnemiesTargetMe().push_back(m_unit);
				}
			}

			else if (!m_unit->isDetected()) {
				for (auto s : INFO.getUnitsInRadius(S, m_unit->getPosition(), m_unit->getType().groundWeapon().maxRange() + 32)) {
					if (abs(atan2(s->pos().y - m_unit->getPosition().y, s->pos().x - m_unit->getPosition().x) - m_unit->getAngle()) < 0.035)
						s->getEnemiesTargetMe().push_back(m_unit);
				}
			}
		}

		// TODO 위치 이동. 불필요하게 여러번 세팅될 수 있음.
		if (!m_enemiesTargetMe.empty())
		{
			m_avgEnemyPosition = UnitUtil::GetAveragePosition(m_enemiesTargetMe);
			m_veryFrontEnemUnit = UnitUtil::GetClosestEnemyTargetingMe(m_unit, m_enemiesTargetMe);
		}
	}
	else {
		// 이전 프레임에 보였다가 갑자기 안보이는 경우
		m_hide = true;

		m_energy = min((double)E->maxEnergy(m_type), m_energy + 0.03125);

		if (m_cooldown > 0) m_cooldown--;

		if (m_lastPosition != Positions::Unknown) {
			// 저그의 burrow 가능한 유닛들이나 vulture 의 spider mine 동작.
			if (m_burrowed)
			{
				// 럴커 주변에 지상유닛이 있는데도 안보이면 없다고 판단.
				if (m_type == Zerg_Lurker)
				{
					if (INFO.getUnitsInRadius(S, m_lastPosition, 5 * TILE_SIZE, true, false, true).size())
					{
						if (m_lastNearUnitFrame + 1 == TIME)
							m_nearUnitFrameCnt++;
						else
							m_nearUnitFrameCnt = 1;

						m_lastNearUnitFrame = TIME;

						if (m_nearUnitFrameCnt > 70) {
							cout << TIME << " : lurker position changed to unknown" << endl;
							m_lastPosition = Positions::Unknown;
						}
					}
				}
			}
			else
			{
				// Visible 아닌 적이 있는 경우에만 Unknown 처리 한다.
				if (bw->isVisible(TilePosition(m_lastPosition)) && bw->isVisible(TilePosition(m_vPosition)))
					m_lastPosition = Positions::Unknown;
			}
		}
	}
}

void UnitInfo::initFrame() {
	if (!m_enemiesTargetMe.empty())
	{
		m_enemiesTargetMe.clear();
		m_avgEnemyPosition = Positions::None;
		m_veryFrontEnemUnit = nullptr;
	}

	if (TIME % 2 == 0)
		m_lastSituation = "";
}

void UnitInfo::setDamage(Unit attacker)
{
	m_expectedDamage += getDamage(attacker, m_unit);
}

PosChange UnitInfo::posChange(UnitInfo *uInfo)
{
	if (uInfo->isHide())
		return PosChange::Stop;

	if (m_lastPosition.getApproxDistance(uInfo->pos()) > m_lastPosition.getApproxDistance(uInfo->vPos()) + 10)
		return PosChange::Closer;

	if (m_lastPosition.getApproxDistance(uInfo->pos()) + 10 < m_lastPosition.getApproxDistance(uInfo->vPos()) )
		return PosChange::Farther;

	return PosChange::Stop;
}

UnitInfo::UnitInfo()
{
}

UnitInfo::~UnitInfo()
{
}
