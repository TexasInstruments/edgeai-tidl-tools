#include <getopt.h>
#include <iostream>
#include <cstdarg>
#include <cstdio>
#include <fstream>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/tidl/tidl_provider_factory.h>
#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>

#include "validator.h"


int main(int argc, char* argv[])
{
    std::string model_path = "";
    std::string image_path = "";
    std::string labels_path = "";
    std::string artifacts_path = "";
    
    int tidl_flag = 0;
    int index;
    int c;
    opterr = 0;

    const char* help_str =
        "Usage: classification_demo <image_path>  <model_path> <artifacts_path> <labels_path> [-t] [-h]\n"
        "Options:\n"
        "    image_path\tPath to the input image to classify\n"
        "    model_path\tPath to the ONNX model\n"
        "    labels_path\tPath to the labels txt file\n"
        "    -t\t\tUse the TIDL execution provider (default CPU execution provider)\n"
        "    -h\t\tDisplay this help text"
        "\n";

    while ((c = getopt (argc, argv, "toh")) != -1)
        switch (c)
        {
        case 't':
            tidl_flag = 1;
            break;
        case 'h':
            fprintf (stdout, help_str, optopt);
            return 0;
        case '?':
            if (isprint (optopt))
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf (stderr,
                         "Unknown option character `\\x%x'.\n",
                         optopt);
            return 1;
        default:
            abort ();
        }

    if ((argc - optind) < 3) {
        fprintf (stderr, help_str, optopt);
        return 1;
    }

    std::cout << argc - optind << std::endl;

    image_path = std::string(argv[optind]);
    model_path = std::string(argv[optind+1]);
    artifacts_path = std::string(argv[optind+2]);
    labels_path = std::string(argv[optind+3]);

    std::cout << image_path << std::endl;
    std::cout << model_path << std::endl;
    std::cout << artifacts_path << std::endl;
    std::cout << labels_path << std::endl;

    for (index = optind + 4; index < argc; index++)
    {
        printf ("!!! Ignoring argument %s\n", argv[index]);
    }

    
    //OrtStatus *status;
    
    // Initialize  enviroment, maintains thread pools and state info
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    
    // Initialize session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    c_api_tidl_options * options = (c_api_tidl_options *)malloc(sizeof(c_api_tidl_options));
    options->debug_level = 0;
    strcpy(options->artifacts_folder, artifacts_path.c_str());

    if (tidl_flag)
    {
        OrtSessionOptionsAppendExecutionProvider_Tidl(session_options, options);
    } 
    else
    {
        OrtSessionOptionsAppendExecutionProvider_CPU(session_options, false);
    }
    
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    // Do the thing
    Validator validator(env, model_path, image_path, labels_path, session_options);

    printf(" Done!\n");
    return 0;
}
