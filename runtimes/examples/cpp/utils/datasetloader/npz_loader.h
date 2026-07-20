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

Redistribution and use in .npz form, without modification, are permitted
provided that the following conditions are met:

*	No reverse engineering, decompilation, or disassembly of this software is
    permitted with respect to any software provided in .npz form.

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

#ifndef NPZ_LOADER_H
#define NPZ_LOADER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <sys/stat.h>

#include "dataset_loader_base.h"
#include "cnpy.h"

/**
 * @brief Loader for .npz files.
 * 
 * Supports continuous loading from a long .npz file with different data types.
 */
class NpzLoader : public DatasetLoaderBase
{
public:
    /**
     * @brief Construct a new NpzLoader object
     * 
     * @param filePath Path to the .npz file
     */
    NpzLoader(const std::string& filePath) : m_filePath(filePath), m_currData(0)
    {
        // Check if file exists
        struct stat buffer;
        if (stat(m_filePath.c_str(), &buffer) != 0)
        {
            throw std::runtime_error("[ERROR] " + m_filePath + " not found");
        }

        // Check if file has .npz extension
        if (m_filePath.substr(m_filePath.find_last_of(".") + 1) != "npz")
        {
            throw std::runtime_error("[ERROR] " + m_filePath + " is not a NPZ file");
        }

        // Load the NPZ file
        m_npzData = cnpy::npz_load(m_filePath);
        
        // Convert to vector for easier access
        for (const auto& item : m_npzData)
        {
            m_dataValues.push_back(&item.second);
        }
        
        m_dataCount = m_dataValues.size();
        
        if (m_dataCount == 0)
        {
            throw std::runtime_error("[ERROR] " + m_filePath + " contains no data");
        }
    }

    /**
     * @brief Load data from the NPZ file.
     *
     * @param data Pointer to pre-allocated memory where data will be stored
     * @param numBytes Number of bytes to load (including padding)
     * @param padT Number of rows to pad at the top. Default : 0
     * @param padB Number of rows to pad at the bottom. Default : 0
     * @param padL Number of columns to pad at the left. Default : 0
     * @param padR Number of columns to pad at the right. Default : 0
     * @param name Key to load from the NPZ file. If non-empty and found, loads by name;
     *             otherwise falls back to index-based loading. Default : ""
     * @return size_t Number of bytes loaded
     */
    size_t load(void* data, size_t numBytes, int32_t padT = 0, int32_t padB = 0, int32_t padL = 0, int32_t padR = 0, const std::string& name = "")
    {
        const cnpy::NpyArray* npyArray = nullptr;

        // Normalize name: replace '::' with '__' to match NPZ keys saved on systems
        // where ':' is invalid in filenames (e.g. Windows zip entries).
        auto normalizeName = [](const std::string& s) {
            std::string result = s;
            size_t pos = 0;
            while ((pos = result.find("::", pos)) != std::string::npos)
            {
                result.replace(pos, 2, "__");
                pos += 2;
            }
            return result;
        };

        auto it = !name.empty() ? m_npzData.find(name) : m_npzData.end();
        if (it == m_npzData.end() && !name.empty())
        {
            it = m_npzData.find(normalizeName(name));
        }
        if (it != m_npzData.end())
        {
            npyArray = &it->second;
        }
        else
        {
            if (!name.empty())
            {
                std::cout << "[WARN] Key '" << name << "' not found in NPZ (available:";
                for (const auto& kv : m_npzData) std::cout << " " << kv.first;
                std::cout << "), falling back to index-based loading" << std::endl;
            }
            if (m_currData >= m_dataCount)
            {
                m_currData = 0;
            }
            npyArray = m_dataValues[m_currData];
            m_currData++;
        }

        if (npyArray->shape.size() < 1)
        {
            throw std::runtime_error("[ERROR] Loaded array has no shape");
        }

        // If no padding is requested, perform a simple copy
        if (padT == 0 && padB == 0 && padL == 0 && padR == 0)
        {
            // Check if the size matches
            if (npyArray->num_bytes() != numBytes)
            {
                throw std::runtime_error("[ERROR] Loaded data size " + std::to_string(npyArray->num_bytes()) + 
                                        " does not match expected size " + std::to_string(numBytes));
            }
            
            // Copy the data
            std::memcpy(data, npyArray->data<void>(), numBytes);
        }
        else
        {
            if((padT != 0 || padB != 0) && npyArray->shape.size() < 2)
            {
                throw std::runtime_error("[ERROR] Top or bottom pad requires atleast 2D tensor. Found 1D tensor.");
            }
            
            // Extract dimensions based on tensor shape
            size_t channels = 1;
            size_t height = 1;
            size_t width;
            size_t elemSize = npyArray->word_size;
            
            if (npyArray->shape.size() >= 3)
            {
                channels = npyArray->shape[npyArray->shape.size() - 3];
                height = npyArray->shape[npyArray->shape.size() - 2];
                width = npyArray->shape[npyArray->shape.size() - 1];
            }
            else if (npyArray->shape.size() == 2)
            {
                height = npyArray->shape[npyArray->shape.size() - 2];
                width = npyArray->shape[npyArray->shape.size() - 1];
            }
            else
            {
                width = npyArray->shape[npyArray->shape.size() - 1];
            }

            // Calculate padded dimensions
            size_t paddedWidth = width + padL + padR;
            size_t paddedHeight = height + padT + padB;
            
            // Calculate strides
            size_t srcRowStride = width * elemSize;
            size_t dstRowStride = paddedWidth * elemSize;
            size_t srcChannelStride = height * srcRowStride;
            size_t dstChannelStride = paddedHeight * dstRowStride;
            
            // Calculate expected size with padding
            size_t expectedSize = channels * paddedHeight * paddedWidth * elemSize;
            if (numBytes != expectedSize)
            {
                throw std::runtime_error("[ERROR] Expected size with padding " + std::to_string(expectedSize) + 
                                        " does not match provided size " + std::to_string(numBytes));
            }
            
            std::memset(data, 0, numBytes);
            
            uint8_t* srcData = static_cast<uint8_t*>(const_cast<void*>(npyArray->data<void>()));
            uint8_t* dstData = static_cast<uint8_t*>(data);
            
            for (size_t c = 0; c < channels; c++)
            {
                uint8_t* srcChannelData = srcData + c * srcChannelStride;
                uint8_t* dstChannelData = dstData + c * dstChannelStride;
                
                uint8_t* dstRowData = dstChannelData + padT * dstRowStride;
                
                for (size_t h = 0; h < height; h++)
                {
                    std::memcpy(dstRowData + padL * elemSize, srcChannelData + h * srcRowStride, srcRowStride);
                    dstRowData += dstRowStride;
                }
            }
        }

        return numBytes;
    }

    /**
     * @brief Reset the loader to start reading from the beginning of the file again.
     */
    void reset()
    {
        m_currData = 0;
    }

    /**
     * @brief Get the number of data items remaining before wrapping around.
     * 
     * @return size_t Number of data items remaining
     */
    size_t getRemainingItems() const
    {
        return m_dataCount - m_currData;
    }

private:
    std::string m_filePath;                     ///< Path to the .npz file
    cnpy::npz_t m_npzData;                      ///< NPZ data loaded from the file
    std::vector<const cnpy::NpyArray*> m_dataValues; ///< Vector of pointers to NpyArray objects
    size_t m_dataCount;                         ///< Number of arrays in the NPZ file
    size_t m_currData;                          ///< Current position in the data values
};

#endif // NPZ_LOADER_H
