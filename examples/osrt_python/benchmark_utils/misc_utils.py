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
import yaml
from .params_base import ParamsBase


def dict_update(src_dict, *args, inplace=False, **kwargs):
    new_dict = src_dict if inplace else src_dict.copy()
    for arg in args:
        assert isinstance(arg, dict), 'arguments must be dict or keywords'
        new_dict.update(arg)
    #
    new_dict.update(kwargs)
    return new_dict


def dict_update_cond(src_dict, *args, inplace=False, condition_fn=None, **kwargs):
    condition_fn = condition_fn if condition_fn is not None else lambda x: (x is not None)
    def _update_conditional(new_dict, arg):
        conditional_arg = {k: v for k,v in arg.items() if condition_fn(v)}
        new_dict.update(conditional_arg)
    #
    new_dict = src_dict if inplace else src_dict.copy()
    for arg in args:
        assert isinstance(arg, dict), 'arguments must be dict or keywords'
        _update_conditional(new_dict, arg)
    #
    _update_conditional(new_dict, kwargs)
    return new_dict


def dict_merge(target_dict, src_dict, inplace=False):
    target_dict = target_dict if inplace else target_dict.copy()
    assert isinstance(target_dict, dict), 'destination must be a dict'
    assert isinstance(src_dict, dict), 'source must be a dict'
    for key, value in src_dict.items():
        if hasattr(target_dict, key) and isinstance(target_dict[key], dict):
            if isinstance(value, dict):
                target_dict[key] = dict_merge(target_dict[key], **value)
            else:
                target_dict[key] = value
            #
        else:
            target_dict[key] = value
        #
    #
    return target_dict


def dict_equal(self, shape1, shape2):
    for k1, v1 in shape1.items():
        if k1 not in shape2:
            return False
        #
        v2 = shape2[k1]
        if isinstance(v1, (list,tuple)) or isinstance(v2, (list,tuple)):
            if any(v1 != v2):
                return False
            #
        elif v1 != v2:
            return False
        #
    #
    return True


def sorted_dict(d, sort_by_value=False):
    if sort_by_value:
        ds = {k:d[k] for k in sorted(d.values())}
    else:
        ds = {k:d[k] for k in sorted(d.keys())}
    #
    return ds


def as_tuple(arg):
    return arg if isinstance(arg, tuple) else (arg,)


def as_list(arg):
    return arg if isinstance(arg, list) else [arg]


def as_list_or_tuple(arg):
    return arg if isinstance(arg, (list,tuple)) else (arg,)


# convert to something that can be saved by yaml.safe_dump
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


def str_to_dict(v):
    if v is None:
        return None
    #
    if isinstance(v, list):
        v = ' '.join(v)
    #
    d = yaml.safe_load(v)
    return d


def str_to_int(v):
    if v in ('', None, 'None'):
        return None
    else:
        return int(v)


def str_to_bool(v):
    if v is None:
        return False
    elif isinstance(v, str):
        if v.lower() in ('', 'none', 'null', 'false', 'no', '0'):
            return False
        elif v.lower() in ('true', 'yes', '1'):
            return True
        #
    #
    return bool(v)


def int_or_none(v):
    if v is None:
        return None
    elif isinstance(v, str):
        if v.lower() in ('', 'none', 'null', 'false', 'no'):
            return None
        elif v.lower() in ('0',):
            return 0
        elif v.lower() in ('true', 'yes', '1'):
            return 1
        #
    #
    return int(v)


def float_or_none(v):
    if v is None:
        return None
    elif isinstance(v, str):
        if v.lower() in ('', 'none', 'null', 'false', 'no'):
            return None
        elif v.lower() in ('0',):
            return 0.0
        elif v.lower() in ('true', 'yes', '1'):
            return 1.0
        #
    #
    return float(v)


def str_or_none(v):
    if v is None:
        return None
    elif isinstance(v, str):
        if v.lower() in ('', 'none', 'null'):
            return None
        #
    #
    return str(v)


def cleanup_dict(inp_dict, template_dict):
    if template_dict is None:
        return inp_dict
    #
    oup_dict = {}
    for k, v in inp_dict.items():
        if k in template_dict:
            if isinstance(v, dict) and isinstance(template_dict[k], dict):
                v = cleanup_dict(v, template_dict[k])
            #
            oup_dict[k] = v
        #
    #
    return oup_dict
