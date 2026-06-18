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
Module containing gelu layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np

def tidl_convert_tanhgelu_to_erfgelu(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Convert TanhGELU approximation to ErfGELU (exact form).
    
    Converts: 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
    To:       0.5 × x × (1 + erf(x / √2))

    SUPPORTED PATTERNS:

    1. STANDARD TANH-GELU WITH POW NODE
       Pattern: Uses Pow(x, 3) for cubic term calculation
    
    2. MUL-CHAIN CUBIC APPROXIMATION
       Pattern: Uses x * x * x multiplication chain
    
    3. FUSED COEFFICIENT OPTIMIZATION
       Pattern: Pre-multiplied constant (0.044715 × 0.7978 = 0.0356774)
    
    4. SIMPLIFIED NO-CUBIC VARIANT
       Pattern: Omits x³ term for faster approximation
    
    5. REORDERED OPERATION SEQUENCE
       Pattern: Mul(0.5) applied before or after Add(1)
    
    6. CUSTOM COEFFICIENT VARIATIONS
       Pattern: Slightly different constants within tolerance (GELU with 0.045 instead of 0.044715)
    """
    # Constants
    SQRT_2 = 1.4142135381698608
    GELU_CONSTANTS = [0.044715, 0.0356774, 0.7978845, 3.0]  # Signatures / Constants
    TOLERANCE = 0.01
    
    converted = 0
    
    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================
    
    def get_constant_value(tensor):
        """Extract scalar from constant tensor"""
        if not isinstance(tensor, gs.Constant):
            return None
        val = tensor.values
        
        return float(val.flatten()[0]) if hasattr(val, 'flatten') else float(val)
    
    def has_value(node, target, tol=TOLERANCE):
        """Check if node has input matching target value"""
        return any(abs(get_constant_value(inp) - target) < tol 
                   for inp in node.inputs if get_constant_value(inp) is not None)
    
    def is_gelu_constant(value):
        """Check if value matches any GELU signature"""
        return any(abs(value - const) < TOLERANCE for const in GELU_CONSTANTS)
    
    def get_nodes_in_direction(start, direction, max_depth=10):
        """Get connected nodes (direction: 'up' or 'down')"""
        visited, result = set(), []
        
        def traverse(node, depth):
            if depth > max_depth or node.name in visited:
                return
            visited.add(node.name)
            result.append(node)
            
            if direction == 'down':
                for out in node.outputs:
                    for consumer in out.outputs:
                        if isinstance(consumer, gs.Node):
                            traverse(consumer, depth + 1)
            else:  # up
                for inp in node.inputs:
                    if isinstance(inp, gs.Variable):
                        for producer in inp.inputs:
                            if isinstance(producer, gs.Node):
                                traverse(producer, depth + 1)
        
        # Start traversal
        if direction == 'down':
            for out in start.outputs:
                for consumer in out.outputs:
                    if isinstance(consumer, gs.Node):
                        traverse(consumer, 1)
        else:
            for inp in start.inputs:
                if isinstance(inp, gs.Variable):
                    for producer in inp.inputs:
                        if isinstance(producer, gs.Node):
                            traverse(producer, 1)
        
        return result
    
    # PATTERN DETECTION (Simplified)
    
    def detect_gelu(tanh_node):
        """Detect if Tanh is part of GELU pattern with mathematical verification"""

        # Step 1: Get ancestors and check for GELU-specific operations
        ancestors = get_nodes_in_direction(tanh_node, 'up', max_depth=8)

        # Check for Mish-specific operations (MUST NOT have these)
        has_log = any(n.op == 'Log' for n in ancestors)
        has_exp = any(n.op == 'Exp' for n in ancestors)

        # If pattern has Log and Exp before Tanh, it's Mish, not GELU
        if has_log and has_exp:
            logging.debug(f"Tanh node {tanh_node.name} is part of Mish activation (has Log+Exp), skipping")
            return None

        # Check for GELU-specific operations (MUST have these)
        has_pow_3 = any(n.op == 'Pow' and has_value(n, 3.0) for n in ancestors)
        has_cubic_mul = False  # Check for x * x * x pattern
        if not has_pow_3:
            # Look for cubic multiplication pattern
            mul_chains = [n for n in ancestors if n.op == 'Mul']
            # Simple heuristic: if there are multiple Mul nodes, might be x*x*x
            has_cubic_mul = len(mul_chains) >= 2

        # Check for GELU signature constants
        has_gelu_coeff = any(has_value(n, 0.044715) for n in ancestors)
        has_tanh_coeff = any(has_value(n, 0.7978845) or has_value(n, 0.7978) for n in ancestors)

        # Must have either Pow(3) or cubic multiplication, AND GELU coefficients
        if not (has_pow_3 or has_cubic_mul) or not (has_gelu_coeff or has_tanh_coeff):
            logging.debug(f"Tanh node {tanh_node.name} missing GELU-specific operations or constants, skipping")
            return None
        
        # Step 2: Find Add(+1) after Tanh
        descendants = get_nodes_in_direction(tanh_node, 'down', max_depth=5)
        add_node = next((n for n in descendants if n.op == 'Add' and has_value(n, 1.0)), None)

        if not add_node:
            return None

        # Step 2.5: Verify 0.5 multiplication exists (GELU-specific, Mish doesn't have this)
        has_half_mul = any(has_value(n, 0.5) for n in descendants)
        if not has_half_mul:
            logging.debug(f"Tanh node {tanh_node.name} missing 0.5 multiplication (not GELU), skipping")
            return None

        # Step 3: Find final Mul - just get all Muls after Add
        add_idx = descendants.index(add_node)
        mul_nodes = [n for n in descendants[add_idx:] if n.op == 'Mul']
        
        if not mul_nodes:
            return None
        
        # Take up to 2 Muls (covers standard and reordered GELU)
        if len(mul_nodes) >= 2:
            final_mul = mul_nodes[1]  # Second Mul (covers both orderings)
        else:
            final_mul = mul_nodes[0]  # Only one Mul 

        # Step 4: Find input variable (most frequently used)
        pattern_nodes = ancestors + [tanh_node] + descendants[:descendants.index(final_mul) + 1]
        var_count = {}
        
        for node in pattern_nodes:
            for inp in node.inputs:
                if isinstance(inp, gs.Variable):
                    var_count[inp.name] = var_count.get(inp.name, 0) + 1
        
        if not var_count:
            return None
        
        input_name = max(var_count, key=var_count.get)
        input_var = next((out for n in graph.nodes for out in n.outputs 
                        if out.name == input_name), None)
        
        if not input_var:
            return None
        
        # Step 5: Collect nodes to remove
        nodes_to_remove = [tanh_node, add_node, final_mul]
        
        # Add intermediate Muls between Add and final_mul
        add_idx = descendants.index(add_node)
        final_idx = descendants.index(final_mul)
        nodes_to_remove.extend([n for n in descendants[add_idx:final_idx] 
                            if n.op == 'Mul' and n not in nodes_to_remove])
        
        # Add GELU constant nodes - ONLY remove Constant nodes, not computation nodes
        for node in ancestors:
            # Skip computation nodes - only remove pure Constant nodes
            if node.op not in ['Constant', 'ConstantOfShape']:
                continue

            constant_values = []
            for inp in node.inputs:
                val = get_constant_value(inp)
                if val is not None:
                    constant_values.append(val)

            has_gelu_const = False
            for constant in constant_values:
                if is_gelu_constant(constant):
                    has_gelu_const = True
                    break

            if has_gelu_const and node not in nodes_to_remove:
                nodes_to_remove.append(node)
        
        return input_var, final_mul.outputs[0], nodes_to_remove
    
    # REPLACEMENT
    def create_erfgelu(input_var, output_var, nodes_to_remove, idx):
        """Create ErfGELU: 0.5 * x * (1 + erf(x / sqrt(2)))"""
        
        prefix = f"erfgelu_{idx}"
        
        # Constants
        c_sqrt2 = gs.Constant(f"{prefix}_sqrt2", np.array(SQRT_2, dtype=np.float32))
        c_one = gs.Constant(f"{prefix}_one", np.array(1.0, dtype=np.float32))
        c_half = gs.Constant(f"{prefix}_half", np.array(0.5, dtype=np.float32))
        
        # Variables
        v_div = gs.Variable(f"{prefix}_div", dtype=np.float32)
        v_erf = gs.Variable(f"{prefix}_erf", dtype=np.float32)
        v_add = gs.Variable(f"{prefix}_add", dtype=np.float32)
        v_mul1 = gs.Variable(f"{prefix}_mul1", dtype=np.float32)
        
        # Output variable
        is_output = output_var in graph.outputs
        v_out = gs.Variable(output_var.name, dtype=output_var.dtype, shape=output_var.shape) if is_output \
                else gs.Variable(f"{prefix}_out", dtype=np.float32)
        
        # Create nodes
        nodes = [
            gs.Node('Div', f"{prefix}_div", {}, [input_var, c_sqrt2], [v_div]),
            gs.Node('Erf', f"{prefix}_erf", {}, [v_div], [v_erf]),
            gs.Node('Add', f"{prefix}_add", {}, [v_erf, c_one], [v_add]),
            gs.Node('Mul', f"{prefix}_mul1", {}, [input_var, v_add], [v_mul1]),
            gs.Node('Mul', f"{prefix}_mul2", {}, [v_mul1, c_half], [v_out])
        ]
        
        # Insert at position of first removed node
        insert_pos = min((graph.nodes.index(n) for n in nodes_to_remove if n in graph.nodes), 
                        default=len(graph.nodes))
        
        for i, node in enumerate(nodes):
            graph.nodes.insert(insert_pos + i, node)
        
        # Redirect consumers
        if is_output:
            graph.outputs = [v_out if o == output_var else o for o in graph.outputs]
        else:
            for consumer in list(output_var.outputs):
                if isinstance(consumer, gs.Node):
                    consumer.inputs = [v_out if inp == output_var else inp 
                                      for inp in consumer.inputs]
        
        # Remove old nodes
        for node in nodes_to_remove:
            if node in graph.nodes:
                graph.nodes.remove(node)
        
        return True
    
    # MAIN LOOP
    for iteration in range(50):
        tanh_nodes = [n for n in graph.nodes if n.op == 'Tanh']

        if not tanh_nodes:
            break

        converted_any = False

        for tanh in tanh_nodes:
            result = detect_gelu(tanh)

            if result:
                input_var, output_var, nodes_to_remove = result
                if create_erfgelu(input_var, output_var, nodes_to_remove, converted):
                    converted += 1
                    converted_any = True
                    logging.debug(f"Converted GELU #{converted}")
                    break

        if not converted_any:
            break

    graph.cleanup()
    graph.toposort()

    logging.debug(f"Total converted: {converted}")


    

def tidl_break_gelu_to_components(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Replace each Gelu node with its mathematical components:
    gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    """
    nodes_to_remove = []
    for node in graph.nodes:
        if node.op == "Gelu":
            x = node.inputs[0]
            logging.info(f"Found Gelu node: {node.name}, replacing with primitive operations.")

            # Create constants
            sqrt2_const = gs.Constant(name=f"{node.name}_sqrt2", values=np.array(1.4142135623730951, dtype=np.float32))
            one_const = gs.Constant(name=f"{node.name}_one", values=np.array(1.0, dtype=np.float32))
            half_const = gs.Constant(name=f"{node.name}_half", values=np.array(0.5, dtype=np.float32))

            # Create intermediate variables
            div_out = gs.Variable(name=f"{node.name}_div_out", dtype=np.float32)
            erf_out = gs.Variable(name=f"{node.name}_erf_out", dtype=np.float32)
            add_out = gs.Variable(name=f"{node.name}_add_out", dtype=np.float32)
            mul1_out = gs.Variable(name=f"{node.name}_mul1_out", dtype=np.float32)

            # Build subgraph with correct wiring
            div_node = gs.Node(op="Div", name=f"{node.name}_div", inputs=[x, sqrt2_const], outputs=[div_out])
            erf_node = gs.Node(op="Erf", name=f"{node.name}_erf", inputs=[div_out], outputs=[erf_out])
            add_node = gs.Node(op="Add", name=f"{node.name}_add", inputs=[erf_out, one_const], outputs=[add_out])
            mul1_node = gs.Node(op="Mul", name=f"{node.name}_mul1", inputs=[x, add_out], outputs=[mul1_out])
            mul2_node = gs.Node(op="Mul", name=f"{node.name}_mul2", inputs=[mul1_out, half_const], outputs=node.outputs)

            logging.debug(f"Adding nodes for Gelu decomposition: {div_node.name}, {erf_node.name}, {add_node.name}, {mul1_node.name}, {mul2_node.name}")

            # Add new nodes to the graph
            graph.nodes.extend([div_node, erf_node, add_node, mul1_node, mul2_node])
            nodes_to_remove.append(node)
            # Disconnect original node
            node.outputs.clear()
            logging.info(f"Replaced Gelu node '{node.name}' with primitive operations.")

    for node in nodes_to_remove:
        logging.info(f"Removing original Gelu node: {node.name}")
        graph.nodes.remove(node)
        