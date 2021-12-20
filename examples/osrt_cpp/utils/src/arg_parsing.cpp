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

#include "../include/arg_parsing.h"

namespace tidl
{
    namespace arg_parsing
    {

        /**
  *  \brief display usage string for application
  * @returns void
  */
        void displayUsage()
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
                << "--device_type, -y: device_type for dlr models can be cpu,gpu\n"
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
        int parseArgs(int argc, char **argv, Settings *s)
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
                    {"device_type", required_argument, nullptr, 'y'},
                    {"labels", required_argument, nullptr, 'l'},
                    {"zoo", required_argument, nullptr, 'z'},
                    {"threads", required_argument, nullptr, 't'},
                    {"num_results", required_argument, nullptr, 'r'},
                    {"warmup_runs", required_argument, nullptr, 'w'},
                    {nullptr, 0, nullptr, 0}};

                /* getopt_long stores the option index here. */
                int option_index = 0;

                c = getopt_long(argc, argv,
                                "v:a:d:c:f:m:i:y:l:t:r:w:z:", long_options,
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
                case 'y':
                    s->device_type = optarg;
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
                    displayUsage();
                    return RETURN_FAIL;;
                default:
                    return RETURN_FAIL;;
                }
            }
            return RETURN_SUCCESS;
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
        void dumpArgs(Settings *s)
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
            std::cout << "device_type set to: " << s->device_type << "\n";
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
