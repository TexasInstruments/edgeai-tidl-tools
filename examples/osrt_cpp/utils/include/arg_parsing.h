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

#ifndef UTILS_ARG_PARSING_H_
#define UTILS_ARG_PARSING_H_

#include <iostream>
#include <vector>
#include <getopt.h>

#include "ti_logger.h"
#include "utility_functs.h"
namespace tidl
{
    namespace arg_parsing
    {
        /**
 @struct  Settings
 @brief   This structure define the parameters of tfl cpp infernce params
*/
        struct Settings
        {
            int log_level = tidl::utils::ERROR;
            bool accel = false;
            bool device_mem = false;
            int loop_count = 1;
            std::vector<float> input_mean;
            std::vector<float> input_std;
            std::string artifact_path = "";
            std::string model_path = "";
            std::string input_bmp_path = "";
            std::string device_type = "cpu";
            std::string labels_file_path = "";
            std::string model_zoo_path = "";
            int number_of_threads = 4;
            int number_of_results = 5;
            int number_of_warmup_runs = 2;
            std::string task_type = "";
        };
        /**
         * Use getopts lib to do options parsing and fill the contents to Setting struct.
         * Will throw error in case of false params.
         *
         *
         * @param args pass the number of args from cmd line
         * @param argv pass the args string  from cmd line
         * @param s pointer to the settings struct to be filled
         *
         * @returns null
         */
        int parseArgs(int argc, char **argv, Settings *s);
        /**
         * Dumps the args set by parsing the arguments use this fnction to display
         * the contents of settings structure. Can be used after parsing the YAML for
         * better clarity of config
         *
         * @param s pointer to the settings struct to be filled
         *
         * @returns null
         */
        void dumpArgs(Settings *s);

    } // arg_parsing
} // tidl

#endif // UTILS_ARG_PARSING_H_