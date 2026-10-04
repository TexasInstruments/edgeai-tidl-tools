"""CPU equivalence and scope checks for the optional MMA Resize workaround."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnx_graphsurgeon as gs
import onnxruntime as ort

from tidl_onnx_model_optimizer import optimize
from tidl_onnx_model_optimizer.ops import get_optimizers, opt_ops
from tidl_onnx_model_optimizer.src.resize import tidl_convert_align_corners_resize_to_matmul


def resize_model(source, target, mode="align_corners", scales=False):
    inputs = ["input", "", "scale"] if scales else ["input", "", "", "size"]
    value = (np.asarray(target, dtype=np.float32) / np.asarray(source, dtype=np.float32)
             if scales else np.asarray(target, dtype=np.int64))
    initializer = numpy_helper.from_array(value, "scale" if scales else "size")
    node = helper.make_node("Resize", inputs, ["output"], name="resize", mode="linear",
                           coordinate_transformation_mode=mode)
    graph = helper.make_graph([node], "resize-test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, source)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, target)], [initializer])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


def lower(model):
    graph = gs.import_onnx(model)
    tidl_convert_align_corners_resize_to_matmul(graph, model.graph)
    graph.cleanup().toposort()
    output = gs.export_onnx(graph)
    output.ir_version = 8
    onnx.checker.check_model(output)
    return output


def evaluate(model, value):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    return session.run(None, {"input": value})[0]


class AlignCornersResizeTests(unittest.TestCase):
    def test_cpu_equivalence_and_border_pixels(self):
        for source, target in (([1, 3, 7, 11], [1, 3, 14, 22]),
                               ([1, 2, 7, 11], [1, 2, 13, 17]),
                               ([1, 2, 7, 11], [1, 2, 3, 5]),
                               ([1, 2, 7, 11], [1, 2, 1, 1]),
                               ([1, 2, 1, 1], [1, 2, 4, 9])):
            with self.subTest(source=source, target=target):
                model = resize_model(source, target)
                converted = lower(copy.deepcopy(model))
                self.assertEqual([n.op_type for n in converted.graph.node].count("MatMul"), 2)
                self.assertNotIn("Resize", [n.op_type for n in converted.graph.node])
                value = np.random.default_rng(53).normal(size=source).astype(np.float32)
                actual, expected = evaluate(converted, value), evaluate(model, value)
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
                np.testing.assert_array_equal(actual[:, :, 0, 0], value[:, :, 0, 0])
                if min(target[-2:]) > 1:
                    np.testing.assert_array_equal(actual[:, :, -1, -1], value[:, :, -1, -1])

    def test_constant_scales_and_output_name(self):
        model = resize_model([1, 2, 7, 11], [1, 2, 14, 22], scales=True)
        converted = lower(model)
        self.assertEqual(converted.graph.output[0].name, "output")
        value = np.random.default_rng(4).normal(size=(1, 2, 7, 11)).astype(np.float32)
        np.testing.assert_allclose(evaluate(converted, value), evaluate(model, value), rtol=2e-5, atol=2e-6)

    def test_other_coordinate_modes_are_unchanged(self):
        for mode in ("half_pixel", "asymmetric"):
            model = resize_model([1, 2, 7, 11], [1, 2, 14, 22], mode=mode)
            self.assertEqual(lower(model).graph.node[0].op_type, "Resize")

    def test_dynamic_or_unsupported_inputs_are_unchanged(self):
        for case in ("dynamic_shape", "dynamic_sizes", "batch", "channel", "dtype", "antialias", "axes"):
            with self.subTest(case=case):
                model = resize_model([1, 2, 7, 11], [1, 2, 14, 22])
                graph = gs.import_onnx(model)
                node = graph.nodes[0]
                if case == "dynamic_shape":
                    node.inputs[0].shape[2] = "height"
                elif case == "dynamic_sizes":
                    sizes = gs.Variable("runtime_sizes", dtype=np.int64, shape=[4])
                    graph.inputs.append(sizes)
                    node.inputs[3] = sizes
                elif case == "batch":
                    node.inputs[0].shape[0] = node.outputs[0].shape[0] = 2
                elif case == "channel":
                    node.outputs[0].shape[1] = 4
                elif case == "dtype":
                    node.inputs[0].dtype = np.float16
                else:
                    node.attrs[case] = 1 if case == "antialias" else [2, 3]
                tidl_convert_align_corners_resize_to_matmul(graph, model.graph)
                self.assertEqual(len(graph.nodes), 1)
                self.assertEqual(graph.nodes[0].op, "Resize")

    def test_generated_names_do_not_collide(self):
        model = resize_model([1, 2, 7, 11], [1, 2, 14, 22])
        graph = gs.import_onnx(model)
        existing = gs.Variable("resize_align_corners_width_input", dtype=np.float32, shape=[1, 2, 7, 11])
        graph.nodes.append(gs.Node(op="Identity", inputs=[graph.inputs[0]], outputs=[existing]))
        graph.outputs.append(existing)
        tidl_convert_align_corners_resize_to_matmul(graph, model.graph)
        converted = gs.export_onnx(graph.cleanup().toposort())
        onnx.checker.check_model(converted)

    def test_registry_and_public_optimizer(self):
        opts = get_optimizers()
        self.assertFalse(opts["convert_align_corners_resize_to_matmul"])
        self.assertIn("convert_align_corners_resize_to_matmul", opt_ops)
        opts = dict.fromkeys(opts, False)
        opts["convert_align_corners_resize_to_matmul"] = True
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "input.onnx", Path(tmp) / "output.onnx"
            onnx.save(resize_model([1, 2, 7, 11], [1, 2, 14, 22]), source)
            optimize(str(source), str(target), custom_optimizers=opts, simplify_mode=None)
            self.assertNotIn("Resize", [n.op_type for n in onnx.load(target).graph.node])


if __name__ == "__main__":
    unittest.main()
