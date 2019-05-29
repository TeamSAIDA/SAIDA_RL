/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "Gym.h"

using namespace BWML;

void Gym::initialize()
{
	// 최초 initializing
	if (!isInitialized) {
		receiveMessage();

		initResMsg->Clear();
		initResMsg = initResMsg->New();
		makeInitMessage(initResMsg);
		sendMessage("Init", initResMsg);
		setInitialzing();

		receiveMessage();

		setResetting();
	}
	// restartGame 시 호출.
	else {
		setResetting();
		doReset();
	}
}

void Gym::doReset(bool isFirstResetCall)
{
	setResetting();
	reset(isFirstResetCall);
}

void Gym::parseInitReq(char *message, int length) {
	initReqMsg->Clear();
	initReqMsg = initReqMsg->New();
	initReqMsg->ParseFromArray(message, length);
	init(initReqMsg);
}

void Gym::parseStepReq(char *message, int length)
{
	stepReqMsg->Clear();
	stepReqMsg = stepReqMsg->New();
	stepReqMsg->ParseFromArray(message, length);
	setActing();
	step(stepReqMsg);
}

void Gym::parseRenderReq(char *message, int length)
{
	renderReqMsg->Clear();
	renderReqMsg = renderReqMsg->New();
	renderReqMsg->ParseFromArray(message, length);
	setRenderData(renderReqMsg);
	sendMessage("Render", renderResMsg);
}

void Gym::updateManager()
{
	stepNum++;
	render();

	if (Config::Debug::Console_Log)
		cout << "(" << stepNum << ") initializeAndValidate" << endl;

	if (!initializeAndValidate())
		return;

	// 1. initialize 중이면 skip
	if (isResetting) {
		if (!isResetFinished()) {
			if (Config::Debug::Console_Log)
				cout << "isResetting" << endl;

			doReset(false);

			return;
		}

		if (Config::Debug::Console_Log)
			cout << "ResetFinished" << endl;

		resetResMsg->Clear();
		resetResMsg = resetResMsg->New();
		makeResetResultMessage(resetResMsg);
		sendMessage("Reset", resetResMsg);
		isResetting = false;
		isActing = false;
		startFrame = TIME;
		episodeNum++;
		stepNum = 0;
	}
	// 2. action 중이면 skip
	else if (isActing) {
		if (!isActionFinished()) {
			if (Config::Debug::Console_Log)
				cout << "isActing" << endl;

			return;
		}

		if (Config::Debug::Console_Log)
			cout << "ActionFinished" << endl;

		stepResMsg->Clear();
		stepResMsg = stepResMsg->New();
		makeStepResultMessage(stepResMsg);
		sendMessage("Step", stepResMsg);
		isActing = false;
	}

	if (Config::Debug::Console_Log)
		cout << "receiveMessage" << endl;

	// 3. Idle인 경우 통신 (step 결과파일 보내고 응답받기)
	receiveMessage();
}

void Gym::makeResetResultMessage(::google::protobuf::Message *message) {
	Message::ResetRes *resetMessage = (Message::ResetRes *)message;

	getObservation(resetMessage->mutable_next_state());
}

void Gym::makeStepResultMessage(::google::protobuf::Message *message) {
	Message::StepRes *stepMessage = (Message::StepRes *)message;

	stepMessage->set_done(isDone());
	stepMessage->set_reward(getReward());
	getInformation(stepMessage->mutable_info());
	getObservation(stepMessage->mutable_next_state());
}

void Gym::getInformation(::google::protobuf::Message *infoMsg) {
	Message::Info *infoMessage = (Message::Info *)infoMsg;

	for (bool ic : isInvalidAction)
		infoMessage->add_was_invalid_action(ic);
}
