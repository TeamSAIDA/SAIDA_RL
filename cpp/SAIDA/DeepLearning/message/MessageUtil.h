#pragma once
#include "common.pb.h"
#include "../../UnitData/UnitInfo.h"

namespace BWML {
	Message::UInfo *setUInfo(Message::UInfo *uInfo, Unit data);
	Message::UInfo *setUInfo(Message::UInfo *uInfo, MyBot::UnitInfo *data);
	Message::TypeInfoMap *setTypeInfo(Message::TypeInfoMap *typeInfoMap, UnitType type);
}