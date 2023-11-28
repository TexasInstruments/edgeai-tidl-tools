# Copyright (c) {2015 - 2023} Texas Instruments Incorporated
#
# All rights reserved not granted herein.
#
# Limited License.
#
# Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
# license under copyrights and patents it now or hereafter owns or controls to make,
# have made, use, import, offer to sell and sell ("Utilize") this software subject to the
# terms herein.  With respect to the foregoing patent license, such license is granted
# solely to the extent that any such patent is necessary to Utilize the software alone.
# The patent license shall not apply to any combinations which include this software,
# other than combinations with devices manufactured by or for TI ("TI Devices").
# No hardware patent is licensed hereunder.
#
# Redistributions must preserve existing copyright notices and reproduce this license
# (including the above copyright notice and the disclaimer and (if applicable) source
# code license limitations below) in the documentation and/or other materials provided
# with the distribution
#
# Redistribution and use in binary form, without modification, are permitted provided
# that the following conditions are met:
#
# *       No reverse engineering, decompilation, or disassembly of this software is
# permitted with respect to any software provided in binary form.
#
# *       any redistribution and use are licensed by TI for use only with TI Devices.
#
# *       Nothing shall obligate TI to provide you with source code for the software
# licensed and provided to you in object code.
#
# If software source code is provided to you, modification and redistribution of the
# source code are permitted provided that the following conditions are met:
#
# *       any redistribution and use of the source code, including any resulting derivative
# works, are licensed by TI for use only with TI Devices.
#
# *       any redistribution and use of any object code compiled from the source code
# and any resulting derivative works, are licensed by TI for use only with TI Devices.
#
# Neither the name of Texas Instruments Incorporated nor the names of its suppliers
#
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# DISCLAIMER.
#
# THIS SOFTWARE IS PROVIDED BY TI AND TI'S LICENSORS "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL TI AND TI'S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.


import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-t', '--traceDir',
                    default='/tmp')
parser.add_argument('-m_golden', '--mapping_golden_file', default='')
parser.add_argument('-m', '--mapping_file', default='')
parser.add_argument('-n', '--numCores', default='4')
parser.add_argument('-s', '--coreStartIdx', default='0')
parser.add_argument('-c','--layerIdxToSkip', default= ['-1'],nargs='+')
args = parser.parse_args()

BATCH_IDX    = 0
DIM1_IDX     = 1
DIM2_IDX     = 2
CHANNEL_IDX  = 3
WIDTH_IDX    = 4
HEIGHT_IDX   = 5
FILENAME_IDX = 6
FOUND_IDX    = 7
IS_FLOAT_CHECK_IDX = 8



mapping_layer_to_dataId_golden = {};
mapping_dataId_to_layer_golden = {};
mapping_layer_to_dataId = {};
mapping_dataId_to_layer = {};


def createMappingDictionary(filename, dictionaryLayerToDataId, dictionaryDataIdToLayer):
   file = open(filename, 'r')
   allLines = file.readlines()
   
   for lines in allLines:
      tempString = lines.split(" ")
      dictionaryLayerToDataId[tempString[2]] = int(tempString[0])
      dictionaryDataIdToLayer[int(tempString[0])] = tempString[2]

   return 

def getFileNameFromDataId(dataId, coreId, numberofskipLayer):   
   file_path = os.path.join(args.traceDir)
   found = 0
   batch = 0
   dim1 = 0
   dim2 = 0
   channel = 0
   width = 0
   height = 0
   filename = 0
   isFloat = 0
   for filename in os.listdir(file_path):      
      if (1):         
         tempFileName = os.path.splitext(filename)[0]         
         splitString = tempFileName.split("_")  
         if(splitString[0] == "C7x" and int(splitString[1]) == (coreId + int(args.coreStartIdx) + 1)):
            isFloat = 0
            if("float" in splitString[-1]):
               baseIdx = -1
               isFloat = 1
            else:
               baseIdx = 0
            dataIdInt  = int(splitString[baseIdx-6])
            dataIdStr = str(dataIdInt)


            if ( dataIdInt == dataId ):
               if(dataIdStr in args.layerIdxToSkip):
                  if(coreId == int(args.coreStartIdx)):
                     print("skipped layer", dataIdStr)
                     numberofskipLayer[0] = numberofskipLayer[0] + 1
               else:
                  batch   = int(splitString[baseIdx-5])
                  dim1    = int(splitString[baseIdx-4])
                  dim2    = int(splitString[baseIdx-3])
                  channel = int(splitString[baseIdx-2])
                  width   = int(splitString[baseIdx-1].split("x")[0])
                  height  = int(splitString[baseIdx-1].split("x")[1])
                  found   = 1
                  filename = os.path.join(file_path,filename)
                  break
   return [batch, dim1, dim2, channel, width, height, filename, found, isFloat ]

def isDataIdFoundForAllCores(coreList, numCoresForLayer):
   found = 1
   for coreIdx in range (0, numCoresForLayer):
      if (coreList[coreIdx][FOUND_IDX] == 0):
         found = 0
         break
   return found

# this functions create stitched dimension along with the file name with new dimension
def getStitchedDimensions(coreList, dataId, numCoresForLayer):
   height = 0
   for coreIdx in range (0, numCoresForLayer):
      height += coreList[coreIdx][HEIGHT_IDX]
   
   filename= f'{dataId:04d}'+"_"+f'{coreList[0][BATCH_IDX]:04d}'+"_"+f'{coreList[0][DIM1_IDX]:04d}'+"_"+f'{coreList[0][DIM2_IDX]:05d}'+"_"+f'{coreList[0][CHANNEL_IDX]:05d}'+"_"+f'{coreList[0][WIDTH_IDX]:05d}'+"x"+f'{height:05d}'
   origFilename = coreList[0][FILENAME_IDX]   
   # Index from the end after removing all the dimensions, so if the name was xyz.txt_0017_0001_0001_00001_00256_00080x00025.y then we are 
   # expected to count from end till xyz.txt_
   index = -40
   filename=origFilename[:index] + filename + origFilename[index+len(filename) :]
   filename = filename.replace(("C7x_" + str((int(args.coreStartIdx) + 1)) + "_"),"C7x_1_")
   filename = filename.replace(args.traceDir, stitch_dir)

   return coreList[0][BATCH_IDX], coreList[0][DIM1_IDX], coreList[0][DIM2_IDX], coreList[0][CHANNEL_IDX], coreList[0][WIDTH_IDX], height, filename

# this functions create stitched dimension along with the file name with new dimension based on mapping files
def getStitchedDimensionsNew(coreList, dataId, numCoresForLayer, dataIdGolden):
   height = 0
   for coreIdx in range (0, numCoresForLayer):
      height += coreList[coreIdx][HEIGHT_IDX]
   
   filename= f'{dataIdGolden:04d}'+"_"+f'{coreList[0][BATCH_IDX]:04d}'+"_"+f'{coreList[0][DIM1_IDX]:04d}'+"_"+f'{coreList[0][DIM2_IDX]:05d}'+"_"+f'{coreList[0][CHANNEL_IDX]:05d}'+"_"+f'{coreList[0][WIDTH_IDX]:05d}'+"x"+f'{height:05d}'
   origFilename = coreList[0][FILENAME_IDX]   
   # Index from the end after removing all the dimensions, so if the name was xyz.txt_0017_0001_0001_00001_00256_00080x00025.y then we are 
   # expected to count from end till xyz.txt_
   index = -40
   filename=origFilename[:index] + filename + origFilename[index+len(filename) :]
   filename = filename.replace(("C7x_" + str((int(args.coreStartIdx) + 1)) + "_"),"C7x_1_")
   filename = filename.replace(args.traceDir, stitch_dir)

   return coreList[0][BATCH_IDX], coreList[0][DIM1_IDX], coreList[0][DIM2_IDX], coreList[0][CHANNEL_IDX], coreList[0][WIDTH_IDX], height, filename
##################################################################

if(args.mapping_golden_file != '' and args.mapping_file != ''):
   createMappingDictionary(args.mapping_golden_file, mapping_layer_to_dataId_golden, mapping_dataId_to_layer_golden);
   createMappingDictionary(args.mapping_file, mapping_layer_to_dataId, mapping_dataId_to_layer);
   print(mapping_layer_to_dataId_golden)

numberofskipLayer = [0]
alreadySkipped = []

#create directoty for Stiched Traces
stitch_dir = os.path.join(args.traceDir, 'stitch_traces')
if(not(os.path.isdir(stitch_dir))) : 
   os.mkdir(stitch_dir)

for layer in range (0,1024):
   coreList= []
   numCoresForLayer=0
   for coreIdx in range (0, int(args.numCores)):
      output = getFileNameFromDataId(layer, coreIdx, numberofskipLayer)
      if(output[FOUND_IDX] == 1):
         numCoresForLayer = numCoresForLayer + 1
         coreList.append(output)

      if(args.mapping_golden_file != '' and args.mapping_file != ''):
         if (numCoresForLayer > 0 and isDataIdFoundForAllCores(coreList, numCoresForLayer) == 1):
            layerNameGolden = mapping_dataId_to_layer[layer]
            if mapping_layer_to_dataId_golden.get(layerNameGolden) is not None:      
               dataIdGolden = mapping_layer_to_dataId_golden[layerNameGolden]
            else:
               continue;

         batch, dim1, dim2, channel, width, height, filename  = getStitchedDimensionsNew(coreList, layer, numCoresForLayer, dataIdGolden)
      elif (numCoresForLayer > 0 and isDataIdFoundForAllCores(coreList, numCoresForLayer) == 1):
         batch, dim1, dim2, channel, width, height, filename  = getStitchedDimensions(coreList, layer - numberofskipLayer[0], numCoresForLayer)

   coreDataList = []
   for coreIdx in range (0, numCoresForLayer):   
      if(output[IS_FLOAT_CHECK_IDX] == 1):
         print(coreList[coreIdx][FILENAME_IDX], "Dumping float data")
         data = np.fromfile(coreList[coreIdx][FILENAME_IDX], dtype=np.float32)
      else:
         data = np.fromfile(coreList[coreIdx][FILENAME_IDX], dtype=np.int8)
      if(numCoresForLayer > 1):
         print(filename)
         data = data.reshape( coreList[coreIdx][BATCH_IDX],
                              coreList[coreIdx][DIM1_IDX],
                              coreList[coreIdx][DIM2_IDX],
                              coreList[coreIdx][CHANNEL_IDX],
                              coreList[coreIdx][HEIGHT_IDX],
                              coreList[coreIdx][WIDTH_IDX] )
      coreDataList.append(data)
   
      finalData = coreDataList[0]
   if(numCoresForLayer > 1):
      for coreIdx in range (0, numCoresForLayer - 1):
         finalData = np.concatenate( (finalData, coreDataList[coreIdx + 1]) , axis=4 )
   if(numCoresForLayer > 0):
      print(filename)
      finalData.tofile(filename)
