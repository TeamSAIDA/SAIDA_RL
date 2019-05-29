/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#ifndef SERVERLOGDLL
#include "MyBotModule.h"
#include "DeepLearning\GymFactory.h"

using namespace MyBot;

void reconnect()
{
	while (!BWAPIClient.connect())
	{
		this_thread::sleep_for(chrono::milliseconds{ 1000 });
		BWAPIClient.update();
	}
}

/// 봇 프로그램을 EXE 형태로 실행합니다
int main(int argc, const char *argv[])
{
	Config::Files::initialize();

#if GYM
	BWML::GymFactory::Instance().Initialize();
#endif

	// 디버그는 cout 을 이용해서 합니다
	cout << "Connecting..." << endl;;
	reconnect();

	cout << "Waiting for a match..." << endl;

	while (Config::BWAPIOptions::RestartGame) {
		while (!Broodwar->isInGame())
		{
			BWAPIClient.update();

			if (!BWAPIClient.isConnected())
			{
				cout << "Reconnecting..." << endl;;
				reconnect();
			}
		}

		Config::BWAPIOptions::RestartGame = false;

#if GYM
		MyBot::FileUtil::eraseHeader(Config::Files::bwapi_ini_filename, "; Start SAIDA_Gym Config", "; End SAIDA_Gym Config");
#endif

		cout << "Match started" << endl;

		MyBotModule *myBotModule = new MyBotModule();

		while (Broodwar->isInGame() && !Broodwar->isReplay())
		{
			for (auto &e : Broodwar->getEvents())
			{
				Logger::debugFrameStr("e : %d\t", e.getType());

				switch (e.getType())
				{
					case EventType::MatchStart:
						myBotModule->onStart();
						break;

					case EventType::MatchEnd:
						myBotModule->onEnd(e.isWinner());
						break;

					case EventType::MatchFrame:
						if (!Config::BWAPIOptions::RestartGame && !Config::BWAPIOptions::EndGame)
							myBotModule->onFrame();

						break;

					case EventType::SendText:
						myBotModule->onSendText(e.getText());
						break;

					case EventType::ReceiveText:
						myBotModule->onReceiveText(e.getPlayer(), e.getText());
						break;

					case EventType::PlayerLeft:
						myBotModule->onPlayerLeft(e.getPlayer());
						break;

					case EventType::NukeDetect:
						myBotModule->onNukeDetect(e.getPosition());
						break;

					case EventType::UnitCreate:
						myBotModule->onUnitCreate(e.getUnit());
						break;

					case EventType::UnitDestroy:
						myBotModule->onUnitDestroy(e.getUnit());
						break;

					case EventType::UnitMorph:
						myBotModule->onUnitMorph(e.getUnit());
						break;

					case EventType::UnitShow:
						myBotModule->onUnitShow(e.getUnit());
						break;

					case EventType::UnitHide:
						myBotModule->onUnitHide(e.getUnit());
						break;

					case EventType::UnitComplete :
						myBotModule->onUnitComplete(e.getUnit());
						break;

					case EventType::UnitDiscover:
						myBotModule->onUnitDiscover(e.getUnit());
						break;

					case EventType::UnitEvade:
						myBotModule->onUnitEvade(e.getUnit());
						break;

					case EventType::UnitRenegade:
						myBotModule->onUnitRenegade(e.getUnit());
						break;

					case EventType::SaveGame:
						myBotModule->onSaveGame(e.getText());
						break;
				}

				Logger::debug("\n");
			}

			if (!BWAPIClient.isConnected())
			{
				cout << "Reconnecting..." << endl;
				reconnect();
			}

			BWAPIClient.update();
		}

		delete myBotModule;
	}

	cout << "Match ended" << endl;

#if GYM
	BWML::GymFactory::Instance().Destroy();
#endif

	return 0;
}
#endif