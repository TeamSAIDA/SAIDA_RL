/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once

#include <thread>
#include <chrono>

#include "Common.h"
#include "CommandUtil.h"
#include "GameCommander.h"
#include "UXManager.h"

#include <BWAPI/Client.h>

namespace MyBot
{
	class MyBotModule : public BWAPI::AIModule
	{
		/// 실제 봇프로그램<br>
		/// @see GameCommander
		GameCommander   gameCommander;

		/// 사용자가 입력한 text 를 parse 해서 처리합니다
		void ParseTextCommand(const string &commandLine);
		_se_translator_function original;
		streambuf *originalStream;

	public:
		MyBotModule();
		~MyBotModule();

		/// 경기가 시작될 때 일회적으로 발생하는 이벤트를 처리합니다
		void onStart();
		///  경기가 종료될 때 일회적으로 발생하는 이벤트를 처리합니다
		void onEnd(bool isWinner);
		/// 경기 진행 중 매 프레임마다 발생하는 이벤트를 처리합니다
		void onFrame();

		/// 유닛(건물/지상유닛/공중유닛)이 Create 될 때 발생하는 이벤트를 처리합니다
		void onUnitCreate(BWAPI::Unit unit);
		///  유닛(건물/지상유닛/공중유닛)이 Destroy 될 때 발생하는 이벤트를 처리합니다
		void onUnitDestroy(BWAPI::Unit unit);

		/// 유닛(건물/지상유닛/공중유닛)이 Morph 될 때 발생하는 이벤트를 처리합니다
		/// Zerg 종족의 유닛은 건물 건설이나 지상유닛/공중유닛 생산에서 거의 대부분 Morph 형태로 진행됩니다
		void onUnitMorph(BWAPI::Unit unit);

		/// 유닛(건물/지상유닛/공중유닛)의 소속 플레이어가 바뀔 때 발생하는 이벤트를 처리합니다.<br>
		/// Gas Geyser에 어떤 플레이어가 Refinery 건물을 건설했을 때, Refinery 건물이 파괴되었을 때, Protoss 종족 Dark Archon 의 Mind Control 에 의해 소속 플레이어가 바뀔 때 발생합니다
		void onUnitRenegade(BWAPI::Unit unit);
		/// 유닛(건물/지상유닛/공중유닛)의 하던 일 (건물 건설, 업그레이드, 지상유닛 훈련 등)이 끝났을 때 발생하는 이벤트를 처리합니다
		void onUnitComplete(BWAPI::Unit unit);

		/// 유닛(건물/지상유닛/공중유닛)이 Discover 될 때 발생하는 이벤트를 처리합니다.<br>
		/// 아군 유닛이 Create 되었을 때 라든가, 적군 유닛이 Discover 되었을 때 발생합니다
		void onUnitDiscover(BWAPI::Unit unit);
		/// 유닛(건물/지상유닛/공중유닛)이 Evade 될 때 발생하는 이벤트를 처리합니다.<br>
		/// 유닛이 Destroy 될 때 발생합니다
		void onUnitEvade(BWAPI::Unit unit);

		/// 유닛(건물/지상유닛/공중유닛)이 Show 될 때 발생하는 이벤트를 처리합니다.<br>
		/// 아군 유닛이 Create 되었을 때 라든가, 적군 유닛이 Discover 되었을 때 발생합니다
		void onUnitShow(BWAPI::Unit unit);
		/// 유닛(건물/지상유닛/공중유닛)이 Hide 될 때 발생하는 이벤트를 처리합니다.<br>
		/// 보이던 유닛이 Hide 될 때 발생합니다
		void onUnitHide(BWAPI::Unit unit);

		/// 핵미사일 발사가 감지되었을 때 발생하는 이벤트를 처리합니다
		void onNukeDetect(BWAPI::Position target);

		/// 다른 플레이어가 대결을 나갔을 때 발생하는 이벤트를 처리합니다
		void onPlayerLeft(BWAPI::Player player);
		/// 게임을 저장할 때 발생하는 이벤트를 처리합니다
		void onSaveGame(string gameName);

		/// 텍스트를 입력 후 엔터를 하여 다른 플레이어들에게 텍스트를 전달하려 할 때 발생하는 이벤트를 처리합니다
		void onSendText(string text);
		/// 다른 플레이어로부터 텍스트를 전달받았을 때 발생하는 이벤트를 처리합니다
		void onReceiveText(BWAPI::Player player, string text);
	};
}