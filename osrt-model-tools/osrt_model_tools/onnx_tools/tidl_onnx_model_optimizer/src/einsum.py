# Copyright (c) {2024 - 2024} Texas Instruments Incorporated
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
"""
Module containing Einsum layer specific functions and optimizations

The Einsum operator is a flexible operator that can represent various matrix operations
through Einstein summation notation. This module provides transformations to replace
Einsum operations with more basic operations like MatMul, Transpose, and Reshape
for better compatibility with TIDL.
"""

import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np
from typing import List, Optional, Dict, Tuple


def parse_operand(operand_str: str) -> Tuple[bool, List[str]]:
    """
    Parse an operand string into tokens.
    Treats '...' as a SINGLE ellipsis token, not 3 dot chars.

    Args:
        operand_str: e.g. "...pd" or "pq..." or "p...q" or "pqd"

    Returns:
        (has_ellipsis, explicit_chars)
        e.g. ("...pd") -> (True, ['p', 'd'])
        e.g. ("pqd")   -> (False, ['p', 'q', 'd'])
    """
    has_ellipsis = '...' in operand_str

    if has_ellipsis:
        # Remove the ellipsis and get remaining explicit chars
        explicit_str   = operand_str.replace('...', '')
        explicit_chars = list(explicit_str)

        # Validate: no stray dots left
        if '.' in explicit_chars:
            logging.warning(
                f"Operand '{operand_str}' has scattered dots — not supported."
            )
            return None, None
    else:
        # No ellipsis: every char is explicit
        if '.' in operand_str:
            logging.warning(
                f"Operand '{operand_str}' has stray dots — not supported."
            )
            return None, None
        explicit_chars = list(operand_str)

    return has_ellipsis, explicit_chars


def get_ellipsis_position(operand_str: str) -> Optional[str]:
    """
    Returns where the ellipsis sits in the operand string.

    Returns:
        'start'  for "...pq"
        'end'    for "pq..."
        'middle' for "p...q"
        'none'   for "pqd"
        None     if invalid (scattered dots)
    """
    if '...' not in operand_str:
        return 'none'

    pos = operand_str.index('...')
    explicit_before = operand_str[:pos]
    explicit_after  = operand_str[pos + 3:]

    if '.' in explicit_before or '.' in explicit_after:
        return None  # Scattered dots

    if pos == 0:
        return 'start'
    elif pos == len(operand_str) - 3:
        return 'end'
    else:
        return 'middle'


def build_dim_map(
    operand_str: str,
    inp_shape: List[int]
) -> Optional[Dict[str, int]]:
    """
    Build a mapping from explicit dimension label -> shape size.

    '...' covers 0 or more batch dimensions.
    Explicit chars map to the remaining dimensions.

    Args:
        operand_str: e.g. "...pd"
        inp_shape:   e.g. [1, 3600, 512]

    Returns:
        dict like {'p': 3600, 'd': 512} or None on error

    Visual:
        "...pd" + [1, 3600, 512]
         ^^^         ^
          |          └── batch dim (covered by ...)
          └── ellipsis covers shape[0..0]

        p -> shape[-2] = 3600
        d -> shape[-1] = 512
    """
    has_ellipsis, explicit_chars = parse_operand(operand_str)

    if explicit_chars is None:
        return None

    if not inp_shape:
        return {ch: None for ch in explicit_chars}

    num_explicit      = len(explicit_chars)
    num_ellipsis_dims = len(inp_shape) - num_explicit

    if num_ellipsis_dims < 0:
        logging.warning(
            f"Operand '{operand_str}' needs {num_explicit} explicit dims "
            f"but shape {inp_shape} only has {len(inp_shape)} dims."
        )
        return None

    if not has_ellipsis:
        # Direct 1-to-1 mapping
        return {ch: inp_shape[i] for i, ch in enumerate(explicit_chars)}

    ellipsis_pos = get_ellipsis_position(operand_str)
    dim_map      = {}

    if ellipsis_pos == 'start':
        # "...pd": explicit chars at END of shape
        # p -> shape[num_ellipsis_dims + 0]
        # d -> shape[num_ellipsis_dims + 1]
        for rank, ch in enumerate(explicit_chars):
            dim_map[ch] = inp_shape[num_ellipsis_dims + rank]

    elif ellipsis_pos == 'end':
        # "pd...": explicit chars at START of shape
        # p -> shape[0]
        # d -> shape[1]
        for rank, ch in enumerate(explicit_chars):
            dim_map[ch] = inp_shape[rank]

    elif ellipsis_pos == 'middle':
        # "p...d": before from START, after from END
        pos          = operand_str.index('...')
        before_chars = list(operand_str[:pos])
        after_chars  = list(operand_str[pos + 3:])

        for rank, ch in enumerate(before_chars):
            dim_map[ch] = inp_shape[rank]

        for rank, ch in enumerate(after_chars):
            dim_map[ch] = inp_shape[len(inp_shape) - len(after_chars) + rank]

    logging.debug(f"dim_map for '{operand_str}' {inp_shape}: {dim_map}")
    return dim_map


def get_batch_dims(operand_str: str, inp_shape: List[int]) -> List[int]:
    """
    Returns the actual batch dimension sizes covered by '...'.

    Args:
        operand_str: e.g. "...pd"
        inp_shape:   e.g. [1, 3600, 512]

    Returns:
        list of batch dim sizes, e.g. [1]
        or [] if ellipsis covers 0 dims
        or [] if no ellipsis

    Visual:
        "...pd" + [1, 3600, 512] -> [1]         (1 batch dim)
        "...pd" + [2, 8, 3600, 512] -> [2, 8]   (2 batch dims)
        "...pd" + [3600, 512]    -> []           (0 batch dims)
        "pqd"   + [3600, 2, 512] -> []           (no ellipsis)
    """
    if '...' not in operand_str:
        return []

    has_ellipsis, explicit_chars = parse_operand(operand_str)
    if explicit_chars is None:
        return []

    num_explicit      = len(explicit_chars)
    num_ellipsis_dims = len(inp_shape) - num_explicit

    if num_ellipsis_dims <= 0:
        return []  # Ellipsis covers 0 dims

    ellipsis_pos = get_ellipsis_position(operand_str)

    if ellipsis_pos == 'start':
        return list(inp_shape[:num_ellipsis_dims])
    elif ellipsis_pos == 'end':
        return list(inp_shape[num_explicit:])
    elif ellipsis_pos == 'middle':
        pos    = operand_str.index('...')
        before = list(operand_str[:pos])
        after  = list(operand_str[pos + 3:])
        return list(inp_shape[len(before): len(inp_shape) - len(after)])

    return []


def build_transpose_perm(
    operand_str: str,
    inp_shape: List[int],
    target_explicit_order: List[str]
) -> Optional[List[int]]:
    """
    Build a transpose permutation to reorder explicit dims,
    keeping batch (ellipsis) dims at front.

    Args:
        operand_str:          e.g. "...pd"
        inp_shape:            e.g. [1, 3600, 512]
        target_explicit_order: e.g. ['d', 'p']  (desired order)

    Returns:
        perm as shape-level indices, e.g. [0, 2, 1]

    Visual:
        "...pd" shape=[1, 3600, 512]
         batch dims at shape[0]     -> keep at front
         p at shape[1]
         d at shape[2]

        target=['d','p'] means we want [batch, d, p]
        perm = [0, 2, 1]
    """
    has_ellipsis, explicit_chars = parse_operand(operand_str)
    if explicit_chars is None:
        return None

    num_explicit      = len(explicit_chars)
    num_ellipsis_dims = len(inp_shape) - num_explicit if inp_shape else 0
    ellipsis_pos      = get_ellipsis_position(operand_str)

    # Build shape_index for each explicit char
    char_to_shape_idx: Dict[str, int] = {}

    if ellipsis_pos == 'start' or not has_ellipsis:
        # explicit chars start at shape[num_ellipsis_dims]
        for rank, ch in enumerate(explicit_chars):
            char_to_shape_idx[ch] = num_ellipsis_dims + rank

    elif ellipsis_pos == 'end':
        # explicit chars start at shape[0]
        for rank, ch in enumerate(explicit_chars):
            char_to_shape_idx[ch] = rank

    elif ellipsis_pos == 'middle':
        pos          = operand_str.index('...')
        before_chars = list(operand_str[:pos])
        after_chars  = list(operand_str[pos + 3:])
        for rank, ch in enumerate(before_chars):
            char_to_shape_idx[ch] = rank
        for rank, ch in enumerate(after_chars):
            char_to_shape_idx[ch] = (
                len(inp_shape) - len(after_chars) + rank
            )

    # Batch dims at the front (always preserved in order)
    if ellipsis_pos == 'start':
        batch_indices = list(range(num_ellipsis_dims))
    elif ellipsis_pos == 'end':
        batch_indices = list(range(num_explicit, len(inp_shape)))
    elif ellipsis_pos == 'middle':
        pos    = operand_str.index('...')
        before = list(operand_str[:pos])
        after  = list(operand_str[pos + 3:])
        batch_indices = list(
            range(len(before), len(inp_shape) - len(after))
        )
    else:
        batch_indices = []

    # Build final perm: batch dims first, then explicit dims in target order
    explicit_perm = [char_to_shape_idx[ch] for ch in target_explicit_order]
    perm          = batch_indices + explicit_perm

    logging.debug(
        f"Transpose perm for '{operand_str}' "
        f"target={target_explicit_order}: {perm}"
    )
    return perm


def tidl_replace_einsum_with_matmul_and_basic_ops(
    graph: gs.Graph,
    onnx_graph: onnx.GraphProto
):
    """
    Replaces Einsum operations with Reshape + Transpose + MatMul.

    Correctly handles '...' as a SINGLE ellipsis covering 0 or more
    batch dimensions (not as individual dot characters).

    Supported:
      "bd,dn->bn"            no ellipsis
      "...pd,...qd->...pq"   ellipsis at start
      "pd...,qd...->pq..."   ellipsis at end
      "p...d,q...d->p...q"   ellipsis in middle

    Not supported (skip with warning):
      "..p..q,..."           scattered dots
    """
    logging.debug("Starting Einsum -> MatMul replacement")

    STRAIGHT_THROUGH_OPS = []
    ELTWISE_OPS          = ['Add', 'Sub', 'Mul', 'Div']

    einsum_nodes = [node for node in graph.nodes if node.op == "Einsum"]
    logging.debug(f"Found {len(einsum_nodes)} Einsum nodes")

    for einsum_node in einsum_nodes:
        logging.debug(f"Processing: {einsum_node.name}")

        # ---------------------------------------------------------------- #
        # Step 1: Parse equation                                            #
        # ---------------------------------------------------------------- #
        equation = einsum_node.attrs.get("equation", "")
        assert isinstance(equation, str)
        logging.debug(f"Equation: {equation}")

        if '->' not in equation:
            logging.debug(f"Skipping: no '->'")
            continue

        lhs, rhs_str = equation.split('->')
        rhs_str      = rhs_str.strip()

        if not rhs_str:
            logging.debug(f"Skipping: empty RHS")
            continue

        operands_str = [op.strip() for op in lhs.split(',')]

        if len(operands_str) != 2:
            logging.debug(f"Skipping: {len(operands_str)} operands, need 2")
            continue

        op_str_a, op_str_b = operands_str

        # ---------------------------------------------------------------- #
        # Step 2: Validate — no scattered dots anywhere                    #
        # ---------------------------------------------------------------- #
        skip = False
        for s in [op_str_a, op_str_b, rhs_str]:
            pos = get_ellipsis_position(s)
            if pos is None:
                logging.warning(
                    f"Skipping {einsum_node.name}: "
                    f"scattered/invalid dots in '{s}'"
                )
                skip = True
                break
        if skip:
            continue

        # ---------------------------------------------------------------- #
        # Step 3: Parse operands -> (has_ellipsis, explicit_chars)         #
        # ---------------------------------------------------------------- #
        has_ell_a, explicit_a = parse_operand(op_str_a)
        has_ell_b, explicit_b = parse_operand(op_str_b)
        has_ell_r, explicit_r = parse_operand(rhs_str)

        if explicit_a is None or explicit_b is None or explicit_r is None:
            logging.warning(f"Skipping {einsum_node.name}: parse failed")
            continue

        logging.debug(
            f"A: has_ellipsis={has_ell_a}, explicit={explicit_a}\n"
            f"B: has_ellipsis={has_ell_b}, explicit={explicit_b}\n"
            f"R: has_ellipsis={has_ell_r}, explicit={explicit_r}"
        )

        # ---------------------------------------------------------------- #
        # Step 4: Build dims from both inputs                              #
        # ---------------------------------------------------------------- #
        inp1, inp2 = einsum_node.inputs
        shape1     = list(inp1.shape) if inp1.shape else []
        shape2     = list(inp2.shape) if inp2.shape else []

        dim_map_a = build_dim_map(op_str_a, shape1)
        dim_map_b = build_dim_map(op_str_b, shape2)

        if dim_map_a is None or dim_map_b is None:
            logging.warning(f"Skipping {einsum_node.name}: dim_map failed")
            continue

        # Merge both dim maps
        dims = {}
        dims.update(dim_map_a)
        dims.update(dim_map_b)
        logging.debug(f"dims = {dims}")

        # ---------------------------------------------------------------- #
        # Step 5: Classify dimensions (explicit only, no dots)             #
        # ---------------------------------------------------------------- #
        # mul dims: in inputs but NOT in rhs explicit chars
        mul_dims = [
            d for d in dims
            if d not in explicit_r
        ]

        if len(mul_dims) != 1:
            logging.debug(
                f"Skipping: {len(mul_dims)} mul dims, need exactly 1"
            )
            continue

        mul_dim = mul_dims[0]

        if mul_dim not in explicit_a or mul_dim not in explicit_b:
            logging.debug(
                f"Skipping: mul dim '{mul_dim}' not in both operands"
            )
            continue

        same_dims   = [
            d for d in dims
            if d in explicit_a and d in explicit_b and d != mul_dim
        ]
        diff_dims_a = [
            d for d in dims
            if d in explicit_a and d not in explicit_b and d != mul_dim
        ]
        diff_dims_b = [
            d for d in dims
            if d not in explicit_a and d in explicit_b and d != mul_dim
        ]

        logging.debug(
            f"same={same_dims}, diff_a={diff_dims_a}, "
            f"diff_b={diff_dims_b}, mul=[{mul_dim}]"
        )

        # ---------------------------------------------------------------- #
        # Step 6: Validate same dims are at start of both operands         #
        # ---------------------------------------------------------------- #
        same_pos_a = [i for i, d in enumerate(explicit_a) if d in same_dims]
        same_pos_b = [i for i, d in enumerate(explicit_b) if d in same_dims]

        if (same_pos_a != list(range(len(same_dims))) or
                same_pos_b != list(range(len(same_dims)))):
            logging.debug("Skipping: common dims not at start of operands")
            continue

        # Save originals for restore
        _inp1 = einsum_node.inputs[0]
        _inp2 = einsum_node.inputs[1]
        _out  = einsum_node.outputs[0]

        # ---------------------------------------------------------------- #
        # Step 7: Transpose A so layout = [..., same, diff_a, mul]         #
        # ---------------------------------------------------------------- #
        target_a = same_dims + diff_dims_a + [mul_dim]

        if explicit_a != target_a:
            logging.debug(
                f"Transposing A: {explicit_a} -> {target_a}"
            )
            perm_a = build_transpose_perm(op_str_a, shape1, target_a)

            if perm_a is None:
                logging.warning(
                    f"Skipping {einsum_node.name}: cannot build perm for A"
                )
                continue

            tr_out_a = gs.Variable(
                f'{einsum_node.name}_tr1_out',
                inp1.dtype,
                [shape1[p] for p in perm_a] if shape1 else None
            )
            tr_node_a = gs.Node(
                'Transpose',
                f'{einsum_node.name}_tr1',
                dict(perm=perm_a),
                [inp1], [tr_out_a]
            )
            graph.nodes.append(tr_node_a)
            einsum_node.inputs[0] = tr_out_a
            op_str_a   = rhs_str[:rhs_str.index('...')+3] + ''.join(target_a) \
                         if has_ell_a else ''.join(target_a)
            explicit_a = target_a
            inp1       = tr_out_a
            logging.debug(f"Transpose A perm={perm_a}")

        # ---------------------------------------------------------------- #
        # Step 8: Transpose B so layout = [..., same, mul, diff_b]         #
        # ---------------------------------------------------------------- #
        target_b = same_dims + [mul_dim] + diff_dims_b

        if explicit_b != target_b:
            logging.debug(
                f"Transposing B: {explicit_b} -> {target_b}"
            )
            inp2   = einsum_node.inputs[1]
            shape2 = list(inp2.shape) if inp2.shape else []

            perm_b = build_transpose_perm(op_str_b, shape2, target_b)

            if perm_b is None:
                logging.warning(
                    f"Skipping {einsum_node.name}: cannot build perm for B"
                )
                continue

            tr_out_b = gs.Variable(
                f'{einsum_node.name}_tr2_out',
                inp2.dtype,
                [shape2[p] for p in perm_b] if shape2 else None
            )
            tr_node_b = gs.Node(
                'Transpose',
                f'{einsum_node.name}_tr2',
                dict(perm=perm_b),
                [inp2], [tr_out_b]
            )
            graph.nodes.append(tr_node_b)
            einsum_node.inputs[1] = tr_out_b
            explicit_b = target_b
            inp2       = tr_out_b
            logging.debug(f"Transpose B perm={perm_b}")

        # ---------------------------------------------------------------- #
        # Step 9: Determine output explicit order and add transpose        #
        # ---------------------------------------------------------------- #
        # What MatMul naturally produces: same + diff_a + diff_b
        natural_output = same_dims + diff_dims_a + diff_dims_b

        out = einsum_node.outputs[0]

        if natural_output != explicit_r:
            logging.debug(
                f"Output Transpose: {natural_output} -> {explicit_r}"
            )
            out_shape = list(out.shape) if out.shape else None

            try:
                perm_out = [natural_output.index(d) for d in explicit_r]
            except ValueError as e:
                logging.warning(
                    f"Skipping {einsum_node.name}: "
                    f"output perm failed: {e}"
                )
                einsum_node.inputs[0]  = _inp1
                einsum_node.inputs[1]  = _inp2
                einsum_node.outputs[0] = _out
                continue

            trans_in = gs.Variable(
                f'{einsum_node.name}_tr_out',
                out.dtype,
                out_shape
            )
            tr_node = gs.Node(
                'Transpose',
                f'{einsum_node.name}_tr',
                dict(perm=perm_out),
                [trans_in], [out]
            )
            graph.nodes.append(tr_node)
            einsum_node.outputs[0] = trans_in
            out = trans_in
            logging.debug(f"Output transpose perm={perm_out}")

        # ---------------------------------------------------------------- #
        # Step 10: Convert to MatMul                                        #
        # ---------------------------------------------------------------- #
        if len(diff_dims_a) == 1 and len(diff_dims_b) == 1:
            logging.debug(f"Converting {einsum_node.name} -> MatMul")
            einsum_node.name += '_MatMul'
            einsum_node.op    = 'MatMul'
            einsum_node.attrs.clear()
            logging.debug(f"Success: {einsum_node.name}")
        else:
            logging.debug(
                f"Cannot convert to MatMul: "
                f"diff_a={diff_dims_a}, diff_b={diff_dims_b}"
            )
            einsum_node.inputs[0]  = _inp1
            einsum_node.inputs[1]  = _inp2
            einsum_node.outputs[0] = _out

    logging.debug("Einsum -> MatMul replacement complete")


