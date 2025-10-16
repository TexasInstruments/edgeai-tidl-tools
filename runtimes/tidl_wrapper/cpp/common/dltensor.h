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

#ifndef DL_TENSOR_H_
#define DL_TENSOR_H_

#include <iostream>
#include <vector>


namespace dl_tensor
{
    /**
     * @class DlTensor
     * @brief A class representing a tensor for deep learning operations
     * 
     * This class encapsulates all the necessary information about a tensor used in
     * deep learning operations, including its name, type, shape, dimensions, and data.
     * It provides a unified representation that can be used across different frameworks
     * and hardware accelerators.
     */
    class DlTensor
    {
        public:

            /** @brief Name of the tensor */
            const char                  *name;

            /** @brief Unified type mapping to TIDL internal type */
            int32_t                     type;

            /** @brief Type in string format */
            std::string                 typeName;

            /** @brief Total size in bytes of the tensor to be allocated
             * This includes any padding required by the hardware or memory alignment.
             */
            int64_t                     allocSize;

            /** @brief Valid size in bytes of the tensor that can be written
             * This represents the actual usable size of the tensor data, which may be
             * smaller than allocSize
             */
            int64_t                     validSize;

            /** @brief Total number of elements in the tensor
             * This should be equal to the product of all dimensions.
             * The size of the type is not accounted in this.
             */
            int64_t                     numElem;

            /** @brief Element size in bytes */
            int32_t                     elemSize;

            /** @brief Number of dimensions in the tensor */
            int32_t                     numDim;

            /** @brief Shape of the tensor as a vector of dimension sizes */
            std::vector<int64_t>        shape;

            /** @brief Top pad of the tensor */
            int32_t                     padT;

            /** @brief Bottom pad of the tensor */
            int32_t                     padB;

            /** @brief Left pad of the tensor */
            int32_t                     padL;

            /** @brief Right pad of the tensor */
            int32_t                     padR;

            /** @brief Pointer to the tensor's data buffer */
            void                        *data;
        
        public:
            /**
             * @brief Prints detailed information about the tensor
             * 
             * Outputs tensor properties including name, type, number of elements,
             * element size, total size, number of dimensions, and shape.
             * The shape is printed in the format [dim1, dim2, ..., dimN].
             */
            void dumpInfo()
            {
                printf("    Name          = %s\n", name);
                printf("    Type          = %s\n", typeName.c_str());
                printf("    TIDL Type     = %d\n", type);
                printf("    Num Elements  = %ld\n", numElem);
                printf("    Element Size  = %d bytes\n", elemSize);
                printf("    Alloc Size    = %ld bytes\n", allocSize);
                printf("    Valid Size    = %ld bytes\n", validSize);
                printf("    Num Dims      = %d\n", numDim);
                printf("    Shape         = ");
                for (int32_t i = 0; i < numDim; i++)
                {
                    if (i == 0)
                        printf("[");
                
                    if (i != (numDim - 1))
                        printf("%ld, ", shape[i]);
                    else
                        printf("%ld", shape[i]);

                    if (i == (numDim - 1))
                        printf("]");
                    
                }
                printf("\n");
                printf("    Pad Top       = %d\n", padT);
                printf("    Pad Bottom    = %d\n", padB);
                printf("    Pad Left      = %d\n", padL);
                printf("    Pad Right     = %d\n", padR);
            }
    };
} // namespace dl_tensor

#endif //DL_TENSOR_H_
