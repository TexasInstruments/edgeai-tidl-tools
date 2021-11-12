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

#include "../include/arg_parsing.h"

namespace tidl
{
    namespace arg_parsing
    {

        /**
  *  \brief display usage string for application
  * @returns void
  */
        void display_usage()
        {
            std::cout
                << "--verbose, -v: [0|1] print more information\n"
                << "--accelerated, -a: [0|1], use Android NNAPI or not\n"
                << "--dev_mem, -d: [0|1], dev_mem or not\n"
                << "--count, -c: loop interpreter->Invoke() for certain times\n"
                << "--input_mean, -b: input mean\n"
                << "--input_std, -s: input standard deviation\n"
                << "--artifact_path, -f: [0|1], Path for Delegate artifacts folder \n"
                << "--model, -m: model path\n"
                << "--image, -i: input_bmp_name with full path\n"
                << "--labels, -l: labels for the model\n"
                << "--zoo, -z: tidl model-zoo path\n"
                << "--threads, -t: number of threads\n"
                << "--num_results, -r: number of results to show\n"
                << "--warmup_runs, -w: number of warmup runs\n"
                << "\n";
        }

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
        void *parse_args(int argc, char **argv, Settings *s)
        {
            int c;
            while (1)
            {
                static struct option long_options[] = {
                    {"log_level", required_argument, nullptr, 'v'},
                    {"accelerated", required_argument, nullptr, 'a'},
                    {"device_mem", required_argument, nullptr, 'd'},
                    {"count", required_argument, nullptr, 'c'},
                    {"artifact_path", required_argument, nullptr, 'f'},
                    {"model", required_argument, nullptr, 'm'},
                    {"image", required_argument, nullptr, 'i'},
                    {"labels", required_argument, nullptr, 'l'},
                    {"zoo", required_argument, nullptr, 'z'},
                    {"threads", required_argument, nullptr, 't'},
                    {"num_results", required_argument, nullptr, 'r'},
                    {"warmup_runs", required_argument, nullptr, 'w'},
                    {nullptr, 0, nullptr, 0}};

                /* getopt_long stores the option index here. */
                int option_index = 0;

                c = getopt_long(argc, argv,
                                "v:a:d:c:f:m:i:l:t:r:w:z:", long_options,
                                &option_index);

                /* Detect the end of the options. */
                if (c == -1)
                    break;

                switch (c)
                {
                case 'v':
                    s->log_level = strtol(optarg, nullptr, 10);
                    break;
                case 'a':
                    s->accel = strtol(optarg, nullptr, 10);
                    break;
                case 'd':
                    s->device_mem = strtol(optarg, nullptr, 10);
                    break;
                case 'c':
                    s->loop_count = strtol(optarg, nullptr, 10);
                    break;
                case 'f':
                    s->artifact_path = optarg;
                    break;
                case 'm':
                    s->model_path = optarg;
                    break;
                case 'i':
                    s->input_bmp_path = optarg;
                    break;
                case 'l':
                    s->labels_file_path = optarg;
                    break;
                case 'z':
                    s->model_zoo_path = optarg;
                    break;
                case 't':
                    s->number_of_threads = strtol(optarg, nullptr, 10);
                    break;
                case 'r':
                    s->number_of_results = strtol(optarg, nullptr, 10);
                    break;
                case 'w':
                    s->number_of_warmup_runs = strtol(optarg, nullptr, 10);
                    break;
                case 'h':
                case '?':
                    /* getopt_long already printed an error message. */
                    display_usage();
                    exit(-1);
                default:
                    exit(-1);
                }
            }
            return s;
        }

        /**
 * Dumps the args set by parsing the arguments use this fnction to display
 * the contents of settings structure. Can be used after parsing the YAML for 
 * better clarity of config 
 * 
 * @param s pointer to the settings struct to be filled
 *
 * @returns null
 */
        void *dump_args(Settings *s)
        {
            std::cout << "\n***** Display run Config: start *****\n";

            std::cout << "verbose level set to: " << s->log_level << "\n";
            std::cout << "accelerated mode set to: " << s->accel << "\n";
            std::cout << "device mem set to: " << s->device_mem << "\n";
            std::cout << "accelerated mode set to: " << s->accel << "\n";
            std::cout << "loop count set to: " << s->loop_count << "\n";
            std::cout << "input mean set to: " << s->input_mean.data() << "\n";
            std::cout << "input std set to: " << s->input_std.data() << "\n";
            std::cout << "artifacts path set to: " << s->artifact_path << "\n";
            std::cout << "model path set to: " << s->model_path << "\n";
            std::cout << "image path set to: " << s->input_bmp_path << "\n";
            std::cout << "labels path set to: " << s->labels_file_path << "\n";
            std::cout << "model zoo path set to: " << s->model_zoo_path << "\n";
            std::cout << "num of threads set to: " << s->number_of_threads << "\n";
            std::cout << "num of results set to: " << s->number_of_results << "\n";
            std::cout << "num of warmup runs set to: " << s->number_of_warmup_runs << "\n";
            std::cout << "task type set to: " << s->task_type << "\n";

            std::cout << "\n***** Display run Config: end *****\n";
        }

    } //arg_parsing
} //tidl
