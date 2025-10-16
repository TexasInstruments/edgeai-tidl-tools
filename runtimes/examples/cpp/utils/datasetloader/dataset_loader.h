/*
Copyright (c) 2026 Texas Instruments Incorporated

All rights reserved not granted herein.

Limited License.  

Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
license under copyrights and patents it now or hereafter owns or controls to 
make, have made, use, import, offer to sell and sell ("Utilize") this software
subject to the terms herein.  With respect to the foregoing patent license,
such license is granted  solely to the extent that any such patent is necessary
to Utilize the software alone.  The patent license shall not apply to any
combinations which include this software, other than combinations with devices
manufactured by or for TI ("TI Devices").  No hardware patent is licensed
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

THIS SOFTWARE IS PROVIDED BY TI AND TI'S LICENSORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL TI AND TI'S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef DATASET_LOADER_H
#define DATASET_LOADER_H

#include <string>
#include <memory>
#include <map>
#include <stdexcept>
#include <algorithm>
#include <cctype>

#include "dataset_loader_base.h"
#include "bin_loader.h"
#include "npz_loader.h"

/**
 * @brief Factory class to create appropriate dataset loaders.
 */
class DatasetLoader
{
public:
    /**
     * @brief Create a dataset loader of the specified type.
     * 
     * @param loaderType Type of loader to create (e.g., 'bin', 'npz')
     * @param args Additional arguments for the loader (e.g., file_path)
     * @return std::unique_ptr<DatasetLoaderBase> The created dataset loader
     * @throws std::runtime_error If the loader type is not supported or required arguments are missing
     */
    static std::unique_ptr<DatasetLoaderBase> createLoader(const std::string& loaderType, const std::map<std::string, std::string>& args)
    {
        std::string loaderTypeLower = loaderType;
        std::transform(loaderTypeLower.begin(), loaderTypeLower.end(), loaderTypeLower.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        
        if (loaderTypeLower == "bin")
        {
            if (args.find("file_path") == args.end())
            {
                throw std::runtime_error("Missing required argument 'file_path' for BinLoader");
            }
            return std::make_unique<BinLoader>(args.at("file_path"));
        }
        else if (loaderTypeLower == "npz")
        {
            if (args.find("file_path") == args.end())
            {
                throw std::runtime_error("Missing required argument 'file_path' for NpzLoader");
            }
            return std::make_unique<NpzLoader>(args.at("file_path"));
        }
        else
        {
            throw std::runtime_error("Unsupported loader type: " + loaderType);
        }
    }
};

#endif // DATASET_LOADER_H
