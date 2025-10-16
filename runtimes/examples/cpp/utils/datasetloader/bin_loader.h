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

#ifndef BIN_LOADER_H
#define BIN_LOADER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <sys/stat.h>

#include "dataset_loader_base.h"

/**
 * @brief Loader for binary files.
 * 
 * Supports continuous loading from a long binary file with different data types.
 */
class BinLoader : public DatasetLoaderBase
{
public:
    /**
     * @brief Construct a new BinLoader object
     * 
     * @param filePath Path to the binary file
     */
    BinLoader(const std::string& filePath) : m_filePath(filePath), m_startIdx(0)
    {
        // Check if file exists
        struct stat buffer;
        if (stat(m_filePath.c_str(), &buffer) != 0)
        {
            throw std::runtime_error("[ERROR] " + m_filePath + " not found");
        }

        std::ifstream file(m_filePath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("[ERROR] Failed to open " + m_filePath);
        }

        m_binarySize = file.tellg();
        file.seekg(0, std::ios::beg);

        m_binaryData.resize(m_binarySize);
        file.read(reinterpret_cast<char*>(m_binaryData.data()), m_binarySize);
        file.close();
    }

    /**
     * @brief Load data from the binary file starting from the current position.
     * 
     * This method allows loading a specific number of bytes directly, without
     * needing to specify the shape. The application is responsible for ensuring
     * the correct size is requested.
     * 
     * @param data Pointer to pre-allocated memory where data will be stored
     * @param numBytes Number of bytes to load
     * @return size_t Number of bytes loaded
     */
    size_t load(void* data, size_t numBytes)
    {
        // Check if we have enough bytes left in the binary data
        if ((m_startIdx + numBytes) > m_binarySize)
        {
            throw std::runtime_error("[ERROR] Not enough data left in the binary file. "
                                    "Requested " + std::to_string(numBytes) + 
                                    " bytes starting at position " + std::to_string(m_startIdx) + 
                                    ", but only " + std::to_string(m_binarySize - m_startIdx) + 
                                    " bytes are available.");
        }

        // Copy the data from the binary buffer to the user-provided memory
        std::memcpy(data, m_binaryData.data() + m_startIdx, numBytes);

        // Update the start index for the next load
        m_startIdx += numBytes;

        return numBytes;
    }

    /**
     * @brief Reset the loader to start reading from the beginning of the file again.
     */
    void reset()
    {
        m_startIdx = 0;
    }

    /**
     * @brief Get the number of bytes remaining in the binary file.
     * 
     * @return size_t Number of bytes remaining
     */
    size_t getRemainingBytes() const
    {
        return m_binarySize - m_startIdx;
    }

private:
    std::string m_filePath;             ///< Path to the binary file
    std::vector<uint8_t> m_binaryData;  ///< Buffer holding the binary data
    size_t m_binarySize;                ///< Size of the binary data in bytes
    size_t m_startIdx;                  ///< Current position in the binary data
};

#endif // BIN_LOADER_H
