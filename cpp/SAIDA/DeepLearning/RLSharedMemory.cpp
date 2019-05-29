/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "RLSharedMemory.h"

using namespace BWML;
using namespace MyBot;

char *RLSharedMemory::receiveHandler(char *message)
{
	//cout << "[R]" << message << endl;
	char *context;
	lastOperation = string(strtok_s(message, ";", &context));

	char *len = strtok_s(NULL, ";", &context);

	int messageSize = atoi(len);

	if (lastOperation == "Init") {
		GymFactory::Instance().GetGym()->parseInitReq(context, messageSize);
	}
	else if (lastOperation == "Reset") {
		GymFactory::Instance().GetGym()->doReset();
	}
	else if (lastOperation == "Step") {
		GymFactory::Instance().GetGym()->parseStepReq(context, messageSize);
	}
	else if (lastOperation == "Render") {
		GymFactory::Instance().GetGym()->parseRenderReq(context, messageSize);
		receive();
	}
	else if (lastOperation == "Close") {
		GymFactory::Instance().GetGym()->close();
		leaveGame();
	}
	else if (lastOperation == "Create") {
		cout << lastOperation << endl;
	}

	return nullptr;
}
