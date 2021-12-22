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

#ifndef DLR_MAIN_H_
#define DLR_MAIN_H_

#include <iostream>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <functional>
#include <vector>
#include <limits>
#include <stdexcept>
#include <dlr.h>
#include <libgen.h>
#include <utility>
#include <thread>
#include <chrono>
#include <mutex>
#include <pthread.h>
#include <sys/time.h>
#include <sys/stat.h>

#include <memory.h>
#include <unistd.h>
#include <getopt.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc_c.h"

#include "itidl_rt.h"
#include "post_process/post_process.h"
#include "pre_process/pre_process.h"
#include "utils/include/arg_parsing.h"
#include "utils/include/utility_functs.h"
#include "utils/include/ti_logger.h"
#include "utils/include/model_info.h"

using namespace tidl::arg_parsing;
using namespace tidl::modelInfo;
using namespace tidl::utility_functs;
using namespace tidl::utils;
using namespace tidl::postprocess;
using namespace tidl::preprocess;

#endif //DLR_MAIN_H