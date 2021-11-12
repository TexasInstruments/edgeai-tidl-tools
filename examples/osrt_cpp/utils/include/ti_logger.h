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

//Error reporting in tidl_edge_tools
#define LOG(x) std::cerr 
// ALways have the error reporting
#define LOG_ERROR(msg, ...) logMsg(tidl::utils::ERROR, "[%s:%04d] " msg, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define LOG_ERROR_RAW(msg, ...) logMsgRaw(ERROR, msg, ##__VA_ARGS__)

// Control the other levels for performance reasons
#if !defined(MINIMAL_LOGGING)

#define LOG_DEBUG(msg, ...) logMsg(tidl::utils::DEBUG, "[%s:%04d] " msg, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define LOG_WARN(msg, ...) logMsg(tidl::utils::WARN, "[%s:%04d] " msg, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define LOG_INFO(msg, ...) logMsg(tidl::utils::INFO, "[%s:%04d] " msg, __FUNCTION__, __LINE__, ##__VA_ARGS__)

/* Logs without timestamping. */
#define LOG_DEBUG_RAW(msg, ...) logMsgRaw(tidl::utils::DEBUG, msg, ##__VA_ARGS__)
#define LOG_WARN_RAW(msg, ...) logMsgRaw(tidl::utils::WARN, msg, ##__VA_ARGS__)
#define LOG_INFO_RAW(msg, ...) logMsgRaw(tidl::utils::INFO, msg, ##__VA_ARGS__)

#else // defined(MINIMAL_LOGGING)

#define LOG_DEBUG(msg, ...)
#define LOG_WARN(msg, ...)
#define LOG_INFO(msg, ...)

#define LOG_DEBUG_RAW(msg, ...)
#define LOG_WARN_RAW(msg, ...)
#define LOG_INFO_RAW(msg, ...)

#endif // if-else !defined(MINIMAL_LOGGING)
#endif // _TI_EDGEAI_LOGGER_H_
