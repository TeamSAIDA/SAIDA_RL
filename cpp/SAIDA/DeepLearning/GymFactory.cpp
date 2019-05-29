/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "../Common.h"
#include "GymFactory.h"
#include <BWAPI/Client.h>
#include "GridWorld/RLGridWorld.h"
#include "VultureVsZealot/RLVultureVsZealot.h"
#include "AvoidObservers/RLAvoidObservers.h"
#include "AvoidReavers/RLAvoidReavers.h"
#include "AvoidZerglings/RLAvoidZerglings.h"
#include "MarineVsZergling/RLMarineVsZergling.h"
#include "MarineScvDefence/RLMarineScvDefence.h"
#include "GoliathVsGoliath/RLGoliathVsGoliath.h"
#include "ZealotVsZealot/RLZealotVsZealot.h"
#include "MarineVsZealot/RLMarineVsZealot.h"

using namespace BWML;

// Gym 이 추가되면 변경되어야 할 메소드
void GymFactory::String2Gym(string gymName, string shmName, int version)
{
	string gymNameStr = (string)gymName;

	if (gymNameStr == "GridWorld") {
		gym = &MyBot::RLGridWorld::Instance(shmName);
		mapName = "GridWorld_v0.scx";
	}
	else if (gymNameStr == "VultureVsZealot") {
		gym = &RLVultureVsZealot::Instance(shmName);

		if (version == 0)
			mapName = "VultureVsZealot_v0.scm";
		else if (version == 1)
			mapName = "VultureVsZealot_v1.scm";
		else if (version == 2)
			mapName = "VultureVsZealot_v2.scm";
		else if (version == 3)
			mapName = "VultureVsZealot_v3.scm";
	}
	else if (gymNameStr == "AvoidObservers") {
		gym = &RLAvoidObservers::Instance(shmName);
		mapName = "AvoidObservers_v0.scx";
	}
	else if (gymNameStr == "AvoidReavers") {
		gym = &RLAvoidReavers::Instance(shmName);
		mapName = "AvoidReavers_v0.scx";
	}
	else if (gymNameStr == "AvoidZerglings") {
		gym = &RLAvoidZerglings::Instance(shmName);

		if (version == 0)
			mapName = "AvoidZerglings_v0.scx";
		else if (version == 1) {
			//
			mapName = "AvoidZerglings_v1.scx";
			autoMenuMode = "LAN";
			enemyType = AIType::HUMAN;
		}
		else if (version == 2) {
			mapName = "AvoidZerglings_v1.scx";
			autoMenuMode = "LAN";
			enemyType = AIType::EXE;
			enemyBot = "bwapi-data\\AI\\AvoidZerglings\\SAIDA.exe";
		}
	}
	else if (gymNameStr == "MarineVsZergling") { // marine
		gym = &RLMarineVsZergling::Instance(shmName);
		mapName = "MarineVsZergling_v0.scm";
	}
	else if (gymNameStr == "MarineScvDefence") {
		gym = &RLMarineScvDefence::Instance(shmName);

		if (version == 0)
			mapName = "MarineScvDefence_v0.scm";

		else if (version == 1) {
			mapName = "MarineScvDefence_v1.scm";
			autoMenuMode = "LAN";
			enemyType = AIType::EXE;
			enemyBot = "bwapi-data\\AI\\MarineScvDefence_v1\\SAIDA.exe";
		}
	}
	else if (gymNameStr == "MarineVsZealot") {
		gym = &RLMarineVsZealot::Instance(shmName);
		mapName = "marine3_zealot1.scm";
	}
	else if (gymNameStr == "ZealotVsZealot") {
		gym = &RLZealotVsZealot::Instance(shmName);
		mapName = "zealot3vs3.scm";
	}
	else if (gymNameStr == "GoliathVsGoliath") {
		gym = &RLGoliathVsGoliath::Instance(shmName);
		mapName = "goliath3vs3.scm";
	}
}

GymFactory &BWML::GymFactory::Instance()
{
	static GymFactory instance;
	return instance;
}

void GymFactory::Initialize(ConnMethod method)
{
	if (method == ConnMethod::SHARED_MEMORY) {
		connection = new RLSharedMemory("SAIDA_INIT", 100);
		connection->initialize();
	}

	char gymName[MAX_GYM_NAME_LENGTH];
	char shmName[MAX_GYM_NAME_LENGTH];
	int version = 0;
	bool autoKill = true;
	int random_seed = -1;

	connection->receiveMessage([&gymName, &shmName, &version, &autoKill, &random_seed](char *message) {
		char *context;
		string lastOperation = string(strtok_s(message, ";", &context));

		if (lastOperation == "Init") {
			char *len = strtok_s(NULL, ";", &context);
			char *name = strtok_s(NULL, ";", &context);

			Message::InitReq initReqMsg;
			initReqMsg.ParseFromArray(name, atoi(len));

			if (name != nullptr) {
				string gym_name = initReqMsg.content();
				copy(gym_name.begin(), gym_name.end(), gymName);
				gymName[gym_name.size()] = '\0';

				string shm_name = initReqMsg.content2();
				copy(shm_name.begin(), shm_name.end(), shmName);
				shmName[shm_name.size()] = '\0';

				version = initReqMsg.version();

				autoKill = initReqMsg.auto_kill_starcraft();
				random_seed = initReqMsg.random_seed();

				if (initReqMsg.no_gui())
					Config::BWAPIOptions::EnableGui = false;
				else
					Config::BWAPIOptions::SetLocalSpeed = initReqMsg.local_speed();
			}

			initReqMsg.Clear();
		}
	});

	autoKillStarcraft = autoKill;

	if (autoKillStarcraft)
		MyBot::CommonUtil::killProcessByName("StarCraft.exe");

	String2Gym(gymName, shmName, version);

	MyBot::FileUtil::eraseHeader(Config::Files::bwapi_ini_filename, "; Start SAIDA_Gym Config", "; End SAIDA_Gym Config");

	vector<string> contents;

	contents.push_back("; Start SAIDA_Gym Config");
	contents.push_back("[ai]");
	contents.push_back("ai = ");
	contents.push_back("[auto_menu]");
	contents.push_back("auto_menu = " + autoMenuMode);
	contents.push_back("lan_mode = Local PC");

	char strValue[MAX_PATH];

	GetPrivateProfileString("auto_menu", "map_path", "maps/usemap/", strValue, MAX_PATH, Config::Files::saida_ini_filename);

	contents.push_back("map = " + (string)strValue + mapName);
	contents.push_back("game_type = USE_MAP_SETTINGS");
	contents.push_back("[window]");
	contents.push_back("windowed = ON");
	contents.push_back("width = 640");
	contents.push_back("height = 480");

	if (random_seed >= 0) {
		contents.push_back("[starcraft]");
		contents.push_back("seed_override = " + to_string(random_seed));
	}

	contents.push_back("; End SAIDA_Gym Config");

	MyBot::FileUtil::addHeader(Config::Files::bwapi_ini_filename, contents);

	MyBot::CommonUtil::create_process("injectory.x86.exe", "--launch StarCraft.exe --inject bwapi-data\\BWAPI.dll --set-flags SEM_NOGPFAULTERRORBOX", Config::Files::StarcraftDirectory);

	while (!BWAPIClient.connect())
	{
		this_thread::sleep_for(chrono::milliseconds{ 1000 });
		BWAPIClient.update();
	}

	if (autoMenuMode == "LAN") {
		MyBot::FileUtil::eraseHeader(Config::Files::bwapi_ini_filename, "; Start SAIDA_Gym Config", "; End SAIDA_Gym Config");

		vector<string> contents;

		contents.push_back("; Start SAIDA_Gym Config");
		contents.push_back("[ai]");
		contents.push_back("ai = " + enemyBot);
		contents.push_back("[auto_menu]");
		contents.push_back("auto_menu = " + autoMenuMode);
		contents.push_back("lan_mode = Local PC");
		contents.push_back("map = ");
		contents.push_back("game = JOIN_FIRST");
		contents.push_back("game_type = USE_MAP_SETTINGS");
		contents.push_back("[window]");
		contents.push_back("windowed = ON");
		contents.push_back("width = 640");
		contents.push_back("height = 480");
		contents.push_back("; End SAIDA_Gym Config");

		MyBot::FileUtil::addHeader(Config::Files::bwapi_ini_filename, contents);


		MyBot::CommonUtil::create_process("injectory.x86.exe", "--launch StarCraft.exe --inject bwapi-data\\BWAPI.dll --set-flags SEM_NOGPFAULTERRORBOX", Config::Files::StarcraftDirectory);

		if (enemyType == AIType::EXE)
			MyBot::CommonUtil::create_process((char *)enemyBot.c_str(), NULL, Config::Files::StarcraftDirectory);

		while (!BWAPIClient.connect())
		{
			this_thread::sleep_for(chrono::milliseconds{ 1000 });
			BWAPIClient.update();
		}
	}

	if (connection) {
		cout << "GymFactory init" << endl;
		connection->close();
	}
}

void BWML::GymFactory::Destroy()
{
	if (autoKillStarcraft)
		MyBot::CommonUtil::killProcessByName("StarCraft.exe");
}
