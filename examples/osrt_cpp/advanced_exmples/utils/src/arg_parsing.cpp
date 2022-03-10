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
    namespace arg_parsing_adv
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
                << "--artifact_path, -f: [0|1], Path for Delegate artifacts folder \n"
                << "--images, -i: input images for model in order with full path\n"
                << "--device_type, -y: device_type for dlr models can be cpu,gpu\n"
                << "--pre_empt_delay, -e: pre emmpt delay for models in order\n"
                << "--labels, -l: labels for the model\n"
                << "--model_dirs, -m: model directory for models in order\n"
                << "--priors, -p: thread priority for models in order\n"
                << "--threads, -t: number of threads to be running in parellel\n"
                << "--val_type, -x: validation type 0:prioirty 1:fps calualtion 2:op tensor validation\n"
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
        int parseArgs(int argc, char **argv, tidl::arg_parsing_adv::Settings *s)
        {
            int c;
            int index;
            while (1)
            {
                static struct option long_options[] = {
                    {"log_level", required_argument, nullptr, 'v'},
                    {"accelerated", required_argument, nullptr, 'a'},
                    {"device_mem", required_argument, nullptr, 'd'},
                    {"loop_counts", required_argument, nullptr, 'c'},
                    {"artifact_path", required_argument, nullptr, 'f'},
                    {"images", required_argument, nullptr, 'i'},
                    {"device_type", required_argument, nullptr, 'y'},
                    {"labels", required_argument, nullptr, 'l'},
                    {"model_dirs", required_argument, nullptr, 'm'},
                    {"priorities", required_argument, nullptr, 'p'},
                    {"max_pre_empt_delays", required_argument, nullptr, 'e'},
                    {"threads", required_argument, nullptr, 't'},
                    {"val_type", required_argument, nullptr, 'x'},
                    {nullptr, 0, nullptr, 0}};

                /* getopt_long stores the option index here. */
                int option_index = 0;

                c = getopt_long(argc, argv,
                                "v:a:d:c:f:i:y:l:m:p:e:t:x:", long_options,
                                &option_index);

                /* Detect the end of the options. */
                if (c == -1)
                    break;

                switch (c)
                {
                case 'v':
                    s->log_level = (tidl::utils::LogLevel)strtol(optarg, nullptr, 10);
                    break;
                case 'a':
                    s->accel = strtol(optarg, nullptr, 10);
                    break;
                case 'x':
                    s->validation_type = strtol(optarg, nullptr, 10);
                    break;
                case 'd':
                    s->device_mem = strtol(optarg, nullptr, 10);
                    break;
                case 'c':
                    index = optind - 1;
                    while (index < argc)
                    {

                        if (*argv[index] != '-')
                        { /* check if optarg is next switch */
                            s->loop_counts[index - optind + 1] = strtol(argv[index], nullptr, 10);
                        }
                        else
                            break;
                        index++;
                    }
                    break;
                case 'f':
                    s->artifact_path = optarg;
                    break;
                case 'i':
                    index = optind - 1;
                    while (index < argc)
                    {
                        if (*argv[index] != '-')
                        { /* check if optarg is next switch */
                           s->input_img_paths[index - optind + 1] = argv[index];
                        }
                        else
                            break;
                        index++;
                    }
                    //optind = index - 1;
                    break;
                case 'y':
                    s->device_type = optarg;
                    break;
                case 'l':
                    s->labels_file_path = optarg;
                    break;
                case 'm':
                    index = optind - 1;
                    while (index < argc)
                    {

                        if (*argv[index] != '-')
                        { /* check if optarg is next switch */
                            s->model_paths[index - optind + 1] = argv[index];
                        }
                        else
                            break;
                        index++;
                    }
                    //optind = index - 1;
                    break;
                case 'p':
                    index = optind - 1;
                    while (index < argc)
                    {

                        if (*argv[index] != '-')
                        { /* check if optarg is next switch */
                            s->priors[index - optind + 1] = strtol(argv[index], nullptr, 10);
                        }
                        else
                            break;
                        index++;
                    }
                    break;
                case 'e':
                    index = optind - 1;
                    while (index < argc)
                    {

                        if (*argv[index] != '-')
                        { /* check if optarg is next switch */
                            s->max_pre_empts[index - optind + 1] = atof(argv[index]);
                        }
                        else
                            break;
                        index++;
                    }
                    //optind = index - 1;
                    break;                    
                case 't':
                    s->number_of_threads = strtol(optarg, nullptr, 10);
                    break;
                case 'h':
                case '?':
                    /* getopt_long already printed an error message. */
                    displayUsage();
                    return RETURN_FAIL;
                    ;
                default:
                    return RETURN_FAIL;
                    ;
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
            std::cout << "model1 loop count set to: " << s->loop_counts[0] << "\n";
            std::cout << "model2 loop count set to: " << s->loop_counts[1] << "\n";
            std::cout << "artifacts path set to: " << s->artifact_path << "\n";
            std::cout << "image1 path set to: " << s->input_img_paths[0] << "\n";
            std::cout << "image2 path set to: " << s->input_img_paths[1] << "\n";
            std::cout << "labels path set to: " << s->labels_file_path << "\n";
            std::cout << "model1 path set to: " << s->model_paths[0] << "\n";
            std::cout << "model2 path set to: " << s->model_paths[1] << "\n";
            std::cout << "prior1 set to: " << s->priors[0] << "\n";
            std::cout << "prior2 set to: " << s->priors[1] << "\n";
            std::cout << "maxPreEmt1 set to: " << s->max_pre_empts[0] << "\n";
            std::cout << "maxPreEmt2 set to: " << s->max_pre_empts[1] << "\n";            
            std::cout << "num of threads set to: " << s->number_of_threads << "\n";
            std::cout << "num of results set to: " << s->number_of_results << "\n";
            std::cout << "validateion type set to: " << s->validation_type << "\n";

            std::cout << "\n***** Display run Config: end *****\n";
        }

    } // arg_parsing
} // tidl
