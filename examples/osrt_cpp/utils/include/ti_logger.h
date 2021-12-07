/*
Copyright (c) 2020 – 2021 Texas Instruments Incorporated

All rights reserved not granted herein.

Limited License.

Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
license under copyrights and patents it now or hereafter owns or controls to
make, have made, use, import, offer to sell and sell ("Utilize") this software
subject to the terms herein.  With respect to the foregoing patent license,
such license is granted  solely to the extent that any such patent is necessary
to Utilize the software alone.  The patent license shall not apply to any
combinations which include this software, other than combinations with devices
manufactured by or for TI (“TI Devices”).  No hardware patent is licensed
hereunder.

Redistributions must preserve existing copyright notices and reproduce this
license (including the above copyright notice and the disclaimer and
(if applicable) source code license limitations below) in the documentation
and/or other materials provided with the distribution

Redistribution and use in binary form, without modification, are permitted
provided that the following conditions are met:

*	No reverse engineering, decompilation, or disassembly of this software is
    permitted with respect to any software provided in binary form.

*	any redistribution and use are licensed by TI for use only with TI Devices.

*	Nothing shall obligate TI to provide you with source code for the software
    licensed and provided to you in object code.

If software source code is provided to you, modification and redistribution of
the source code are permitted provided that the following conditions are met:

*	any redistribution and use of the source code, including any resulting
    derivative works, are licensed by TI for use only with TI Devices.

*	any redistribution and use of any object code compiled from the source code
    and any resulting derivative works, are licensed by TI for use only with TI
    Devices.

Neither the name of Texas Instruments Incorporated nor the names of its
suppliers may be used to endorse or promote products derived from this software
without specific prior written permission.

DISCLAIMER.

THIS SOFTWARE IS PROVIDED BY TI AND TI’S LICENSORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL TI AND TI’S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef _TI_EDGEAI_LOGGER_H_
#define _TI_EDGEAI_LOGGER_H_

/* Standard headers. */
#include <stdint.h>

/**
 * \defgroup group_edgeai_utils_logger Logging Utility
 *
 * \brief Multi-level logging utility
 *
 * \ingroup group_edgeai_utils
 */

namespace tidl
{
   namespace utils
   {
      /*! \brief Enumerations for different log levels.
       * \ingroup group_edgeai_utils_logger
       */
      enum LogLevel
      {
         /** Used to show general debug messages */
         DEBUG = 0,

         /** Used to show run-time processing debug */
         INFO = 1,

         /** Used to warning developers of possible issues */
         WARN = 2,

         /** Used for most errors */
         ERROR = 3,

         /** Invalid valie. */
         MAX = 4

      };

      /** Logs the message.
       * \ingroup group_edgeai_utils_logger
       *
       * @param level Log level to use.
       * @param format The format string for printing.
       * @param ... The variable list of arguments
       */
      void logMsg(LogLevel level, const char *format, ...);

      /** Logs the message without timestamp information.
       * \ingroup group_edgeai_utils_logger
       *
       * @param level Log level to use.
       * @param format The format string for printing.
       * @param ... The variable list of arguments
       */
      void logMsgRaw(LogLevel level, const char *format, ...);

      /** Sets a bit in the log level mask.
       * \ingroup group_edgeai_utils_logger
       *
       * @param level Log level to set.
       */
      void logSetLevel(LogLevel level);

   }
} // namespace tidl::utils

/*Error reporting in tidl_edge_tools */
#define LOG(x) std::cerr
/* ALways have the error reporting */
#define LOG_ERROR(msg, ...) logMsg(tidl::utils::ERROR, "[%s:%04d] " msg, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define LOG_ERROR_RAW(msg, ...) logMsgRaw(ERROR, msg, ##__VA_ARGS__)

/* Control the other levels for performance reasons */
#if !defined(MINIMAL_LOGGING)

#define LOG_DEBUG(msg, ...) logMsg(tidl::utils::DEBUG, "[%s:%04d] " msg, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define LOG_WARN(msg, ...) logMsg(tidl::utils::WARN, "[%s:%04d] " msg, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define LOG_INFO(msg, ...) logMsg(tidl::utils::INFO, "[%s:%04d] " msg, __FUNCTION__, __LINE__, ##__VA_ARGS__)

/* Logs without timestamping. */
#define LOG_DEBUG_RAW(msg, ...) logMsgRaw(tidl::utils::DEBUG, msg, ##__VA_ARGS__)
#define LOG_WARN_RAW(msg, ...) logMsgRaw(tidl::utils::WARN, msg, ##__VA_ARGS__)
#define LOG_INFO_RAW(msg, ...) logMsgRaw(tidl::utils::INFO, msg, ##__VA_ARGS__)

#else /* defined(MINIMAL_LOGGING) */

#define LOG_DEBUG(msg, ...)
#define LOG_WARN(msg, ...)
#define LOG_INFO(msg, ...)

#define LOG_DEBUG_RAW(msg, ...)
#define LOG_WARN_RAW(msg, ...)
#define LOG_INFO_RAW(msg, ...)

#endif // if-else !defined(MINIMAL_LOGGING)
#endif // _TI_EDGEAI_LOGGER_H_
