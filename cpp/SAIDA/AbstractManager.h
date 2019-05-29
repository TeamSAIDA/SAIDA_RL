/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once
#include "Common.h"

using namespace std;

namespace MyBot
{
	class AbstractManager
	{
	private:
		void defaultTimeoutLog() {
			Logger::info(Config::Files::TimeoutFilename, true, "%s Timeout (%dms)\n", managerName.c_str(), elapsedTime);
			writeTimeoutLog();
		}

	protected:
		int elapsedTime;
		string managerName;

		virtual void updateManager() = 0;
		virtual void writeSAIDAExceptionLog(SAIDA_Exception e) {
			Logger::error("%s Error. (ErrorCode : %x, Eip : %p)\n", managerName.c_str(), e.getSeNumber(), e.getExceptionPointers()->ContextRecord->Eip);
		}
		virtual void writeExceptionLog(const exception &e) {
			Logger::error("%s Error. (Error : %s)\n", managerName.c_str(), e.what());
		}
		virtual void writeAllExceptionLog() {
			Logger::error("%s Unknown Error.\n", managerName.c_str());
		}
		virtual void writeTimeoutLog() {
			return;
		}

	public:
		AbstractManager(string name) {
			managerName = name;
			elapsedTime = 0;
		}
		virtual ~AbstractManager() {};

		void update() {
			elapsedTime = clock();

			try {
				updateManager();
			}
			catch (SAIDA_Exception e) {
				writeSAIDAExceptionLog(e);
#ifndef NOT_THROW
				throw e;
#endif
			}
			catch (const exception &e) {
				writeExceptionLog(e);
#ifndef NOT_THROW
				throw e;
#endif
			}
			catch (...) {
				writeAllExceptionLog();
#ifndef NOT_THROW
				throw;
#endif
			}

			elapsedTime = clock() - elapsedTime;

			if (elapsedTime > 55) {
				defaultTimeoutLog();
			}
		}

		virtual void onUnitDestroy(Unit unit) {}
		virtual void onUnitCreate(Unit unit) {}
		virtual void onUnitComplete(Unit unit) {}
	};
}
