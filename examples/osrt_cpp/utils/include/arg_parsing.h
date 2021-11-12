/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef UTILS_ARG_PARSING_H_
#define UTILS_ARG_PARSING_H_

#include <iostream>
#include <vector>
#include <getopt.h>


#include "ti_logger.h"
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
            std::string labels_file_path = "";
            std::string model_zoo_path = "";
            int number_of_threads = 4;
            int number_of_results = 5;
            int number_of_warmup_runs = 2;
            std::string task_type = ""; // taken from artifacts.yaml
        };
        /**
 * Use getopts lib to do options parsing and fill the contents to Setting struct.
 * Will throuw error in case of false params.
 * 
 *
 * @param args pass the number of args from cmd line
 * @param argv pass the args string  from cmd line
 * @param s pointer to the settings struct to be filled
 *
 * @returns null
 */
        void *parse_args(int argc, char **argv, Settings *s);
        /**
 * Dumps the args set by parsing the arguments use this fnction to display
 * the contents of settings structure. Can be used after parsing the YAML for 
 * better clarity of config 
 * 
 * @param s pointer to the settings struct to be filled
 *
 * @returns null
 */
        void *dump_args(Settings *s);

    } //arg_parsing
} //tidl

#endif //UTILS_ARG_PARSING_H_