/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once
#include "MessageHandler.h"

class SharedMemoryMessageHandler : public MessageHandler
{
public:
	SharedMemoryMessageHandler();
	~SharedMemoryMessageHandler();
};

