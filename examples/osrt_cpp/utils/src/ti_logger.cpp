/*
 *  Copyright (C) 2021 Texas Instruments Incorporated - http://www.ti.com/
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* Standard headers. */
#include <stdio.h>
#include <stdarg.h>
#include <cstring>
#include <string>
#include <map>
#include <chrono>
#include <sys/time.h>

/* Module headers. */
#include "../include/ti_logger.h"

using namespace std;

namespace tidl
{
    namespace utils
    {

#define VAL_AND_STR(X) \
    {                  \
        X, #X          \
    }

        static char gLogStr[2048];
        static uint32_t gLogLevel = LogLevel::DEBUG;
        static uint64_t gStartTime = 0;

        static map<LogLevel, const char *> gLevelStr =
            {
                VAL_AND_STR(ERROR),
                VAL_AND_STR(WARN),
                VAL_AND_STR(INFO),
                VAL_AND_STR(DEBUG)};

        static uint64_t getTimeInUsecs(struct timeval &tv)
        {
            uint64_t timeInUsecs = 0;

            if (gettimeofday(&tv, NULL) < 0)
            {
                timeInUsecs = 0;
            }
            else
            {
                timeInUsecs = tv.tv_sec * 1000000ull + tv.tv_usec;
            }

            if (gStartTime == 0U)
            {
                gStartTime = timeInUsecs;
            }

            return timeInUsecs - gStartTime;
        }

        void logMsgRaw(LogLevel level, const char *format, ...)
        {
            if (level >= gLogLevel)
            {
                va_list ap;

                va_start(ap, format);

                vsnprintf(gLogStr, sizeof(gLogStr), format, ap);

                gLogStr[sizeof(gLogStr) - 1] = '\0';
                printf("%s", gLogStr);

                va_end(ap);
            }
        }

        void logMsg(LogLevel level, const char *format, ...)
        {
            if (level >= gLogLevel)
            {
                uint64_t curTime;
                uint32_t size;
                uint32_t millisec;
                uint32_t microsec;
                va_list ap;

                va_start(ap, format);

                snprintf(gLogStr, sizeof(gLogStr), "%s:", gLevelStr[level]);
                size = (uint32_t)strlen(gLogStr);
                vsnprintf(&gLogStr[size], sizeof(gLogStr) - size, format, ap);

                struct timeval tv;

                curTime = getTimeInUsecs(tv);
                millisec = curTime / 1000U;
                microsec = curTime % 1000000U;

                struct tm *timeInfo = localtime(&tv.tv_sec);

                printf("[%02i:%02i:%02i.%03i.%06i]:%s",
                       timeInfo->tm_hour, timeInfo->tm_min,
                       timeInfo->tm_sec, millisec, microsec, gLogStr);

                va_end(ap);
            }
        }

        void logSetLevel(LogLevel level)
        {
            if (level < MAX)
            {
                gLogLevel = level;
            }
        }

    }
}

