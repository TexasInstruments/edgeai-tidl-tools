# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np


class ParamsBase:
    def __init__(self):
        self.is_initialized = False

    def initialize(self):
        assert hasattr(self, 'kwargs') and isinstance(self.kwargs, dict), \
            'the child class must have a dict called kwargs'
        self.is_initialized = True

    def get_param(self, param_name):
        assert self.is_initialized, 'initialize must be called before get_param() can be done'
        return self.peek_param(param_name)

    def set_param(self, param_name, value):
        assert hasattr(self, 'kwargs') and isinstance(self.kwargs, dict), \
            'the child class must have a dict called kwargs'
        if hasattr(self, param_name):
            setattr(self, param_name, value)
        elif param_name in self.kwargs:
            self.kwargs[param_name] = value
        else:
            assert False, f'param {param_name} could not be found in object {self.__class__.__name__}'
        #

    def peek_param(self, param_name):
        assert hasattr(self, 'kwargs') and isinstance(self.kwargs, dict), \
            'the child class must have a dict called kwargs'
        # param may not be final yet - use get_param instead to be sure
        if hasattr(self, param_name):
            return getattr(self, param_name)
        elif param_name in self.kwargs:
            return self.kwargs[param_name]
        else:
            assert False, f'param {param_name} could not be found in object {self.__class__.__name__}'
        #

    def get_params(self):
        assert self.is_initialized, 'initialize must be called before get_param() can be done'
        return self.kwargs

    def peek_params(self):
        assert hasattr(self, 'kwargs') and isinstance(self.kwargs, dict), \
            'the child class must have a dict called kwargs'
        return self.kwargs
    
