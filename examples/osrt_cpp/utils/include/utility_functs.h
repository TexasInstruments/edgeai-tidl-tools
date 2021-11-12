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

#ifndef UTILS_UTILITY_FUCTS_H_
#define UTILS_UTILITY_FUCTS_H_


#include <iostream>
#include <vector>
#include <getopt.h>




namespace tidl
{
    namespace utility_functs
    {
        /**
  *  \brief returns time in micro sec
  * @returns void
  */
        double
        get_us(struct timeval t);

    } // namespace utility_functs
} //namespace tidl



#endif //UTILS_UTILITY_FUCTS_H_