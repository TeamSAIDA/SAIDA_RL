/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once
#include "../Common.h"
#include "ConnCommon.h"
#include "MessageHandler.h"
#include "message/common.pb.h"
#include "message/MessageUtil.h"

#define MAX_CHAR_LENGTH 200

namespace BWML {
	class SharedMemory
	{
	private:
		bool connected = false;
		void registerSharedMemory();
		string shmName;
		LPCSTR MapName;

	protected:
		string lastOperation;

		// 공유메모리 생성 후 초기화 할 로직.
		virtual void init() {
			Message::Create create;
			create.set_content("Create Content.");
			send("Create", &create);
		}
		// 받은 데이터를 처리하는 로직
		virtual char *receiveHandler(char *message) {
			return message;
		}
		// 공유메모리 삭제 전 처리해야 할 로직.
		virtual void send_close_message() {
			Message::Close close;
			close.set_content("Close Content.");
			send("Close", &close);
			Sleep(5);
		}

		bool receive();
		template<typename _Pr>
		bool receive(_Pr _Pred) {
			char data[200];
			char *ptr = pData;

			while (ptr && ptr[0] != 'P')
				Sleep(1);

			if (ptr && ptr[0] == 'P') {
				memset(data, NULL, sizeof(data));
				memcpy(data, pData, sizeof(data));

				char *context;
				char *ch = strtok_s(data, ";", &context);

				_Pred(context);
				return true;
			}

			return false;
		}
		void send(char *operation, ::google::protobuf::Message *message);

	public:
		SharedMemory(string name, size_t size)
		{
			hFileMap = 0;
			pData = NULL;
			setMapName(name);
			MAX_SHM_SIZE = size;
		}
		virtual ~SharedMemory() {
			close();
		}

		HANDLE hFileMap;
		char *pData;
		size_t MAX_SHM_SIZE;

		void initialize() {
			registerSharedMemory();
			init();
		}
		bool isConnecting();
		void close();

		virtual bool receiveMessage() {
			return receive();
		}

		template<typename _Pr>
		bool receiveMessage(_Pr _Pred) {
			return receive(_Pred);
		}

		void sendMessage(char *operation, ::google::protobuf::Message *message) {
			send(operation, message);
		}

		string getLastOperation() {
			return lastOperation;
		}

		LPCSTR getMapName() {
			return MapName;
		}

		void setMapName(string mapName) {
			shmName = mapName;
			MapName = LPCSTR(shmName.c_str());
		}
	};
}