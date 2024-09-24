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
    

def pretty_object(d, depth=10, precision=6):
    depth = depth - 1
    pass_through_types = (str, int)
    if depth < 0:
        d_out = None
    elif d is None:
        d_out = d
    elif isinstance(d, pass_through_types):
        d_out = d
    elif isinstance(d, (np.float32, np.float64)):
        # numpy objects cannot be serialized with yaml - convert to float
        d_out = round(float(d), precision)
    elif isinstance(d, np.int64):
        d_out = int(d)
    elif isinstance(d, float):
        # round to the given precision
        d_out = round(d, precision)
    elif isinstance(d, dict):
        d_out = {k: pretty_object(v, depth) for k , v in d.items()}
    elif isinstance(d, (list,tuple)):
        d_out = [pretty_object(di, depth) for di in d]
    elif isinstance(d, np.ndarray):
        d_out = pretty_object(d.tolist(), depth)
    elif isinstance(d, ParamsBase):
        # this is a special case
        p = d.peek_params()
        d_out = pretty_object(p, depth)
    elif hasattr(d, '__dict__'):
        # other unrecognized objects - just grab the attributes as a dict
        attrs = d.__dict__.copy()
        if 'name' not in attrs:
            attrs.update({'name':d.__class__.__name__})
        #
        d_out = pretty_object(attrs, depth)
    else:
        d_out = None
    #
    return d_out