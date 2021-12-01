/*
Copyright (c) 2020 – 2021 Texas Instruments Incorporated

All rights reserved not granted herein.

Limited License.  

Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive license under copyrights and patents it now or hereafter owns or controls to make, have made, use, import, offer to sell and sell ("Utilize") this software subject to the terms herein.  With respect to the foregoing patent license, such license is granted  solely to the extent that any such patent is necessary to Utilize the software alone.  The patent license shall not apply to any combinations which include this software, other than combinations with devices manufactured by or for TI (“TI Devices”).  No hardware patent is licensed hereunder.

Redistributions must preserve existing copyright notices and reproduce this license (including the above copyright notice and the disclaimer and (if applicable) source code license limitations below) in the documentation and/or other materials provided with the distribution

Redistribution and use in binary form, without modification, are permitted provided that the following conditions are met:

*	No reverse engineering, decompilation, or disassembly of this software is permitted with respect to any software provided in binary form.

*	any redistribution and use are licensed by TI for use only with TI Devices.

*	Nothing shall obligate TI to provide you with source code for the software licensed and provided to you in object code.

If software source code is provided to you, modification and redistribution of the source code are permitted provided that the following conditions are met:

*	any redistribution and use of the source code, including any resulting derivative works, are licensed by TI for use only with TI Devices.

*	any redistribution and use of any object code compiled from the source code and any resulting derivative works, are licensed by TI for use only with TI Devices.

Neither the name of Texas Instruments Incorporated nor the names of its suppliers may be used to endorse or promote products derived from this software without specific prior written permission.

DISCLAIMER.

THIS SOFTWARE IS PROVIDED BY TI AND TI’S LICENSORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL TI AND TI’S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

