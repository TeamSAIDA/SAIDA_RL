/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "SharedMemoryManager.h"
#include "SharedMemory.h"

using namespace BWML;

void SharedMemory::registerSharedMemory() {
	connected = SHM.CreateMemoryMap(this);
}

bool SharedMemory::isConnecting()
{
	return connected;
}

bool SharedMemory::receive()
{
	char *ptr = pData;

	while (ptr && ptr[0] != 'P')
		Sleep(1);

	if (ptr && ptr[0] == 'P') {
		receiveHandler(ptr + 2);
		return true;
	}

	return false;
}

void SharedMemory::send(char *operation, ::google::protobuf::Message *message)
{
	int left_size = MAX_SHM_SIZE - 3; // header, tailer 제거한 길이
	char *p = pData + 2;

	char m[20];
	memset(m, NULL, sizeof(m));

	wsprintf(m, "%s;%d;", operation, message->ByteSize());

	strncpy_s(p, left_size, m, left_size);
	int len = strlen(m);
	p += len;
	left_size -= len;

	message->SerializeToArray(p, message->ByteSize());

	//cout << message->ByteSize() << " " << len << m << endl;

	//cout << "[S] " << (operation == nullptr ? "" : operation + string(";")) << message->DebugString() << endl << "S";

	//for (int i = 1; i < len + 2; i++)
	//	printf("%c", *(pData + i));

	//for (int i = 0; i < message->ByteSize(); i++)
	//	printf("%d ", *(p + i));

	//printf("\n");

	// 데이터를 모두 쓴 다음에 플래그를 바꿔준다.
	p = pData;
	*p = 'S';
}

void SharedMemory::close()
{
	if (connected) {
		send_close_message();
		SHM.FreeMemoryMap(this);
		connected = false;
	}
}
