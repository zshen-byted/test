import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.library import Library

from torch._functorch.aot_autograd import aot_module_simplified
from torch._dynamo.backends.common import aot_autograd

import numpy as np
from typing import Optional



class AlignmentNode:
    def __init__(self, id:int):
        self.id = id
        self.meta = {} # {"name": "xxx", "stage": "forward"/"backward", "nn_module_stack": [...], ...}
        self.data = {} # {"model_1": [(xorsum, raw_tensor, ...), ...], "model_2": [(xorsum, raw_tensor, ...), ...], ...}


class AlignmentManager:
    _instance: Optional["AlignmentManager"] = None
    _nodes: dict[int, dict] = {}
    _next_id: int = 0


    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AlignmentManager, cls).__new__(cls)
            cls._register_ops()
        return cls._instance

    def __init__(self):
        pass


    @classmethod
    def _register_ops(cls):

        # ops.xpugraph.marker
        @torch.library.custom_op("xpugraph::marker", mutates_args=())
        def xpugraph_marker(x: torch.Tensor, node_id: int) -> torch.Tensor:
            return x.clone()
        # register fake
        @xpugraph_marker.register_fake
        def xpugraph_marker_fake(x: torch.Tensor, node_id: int) -> torch.Tensor:
            return torch.empty_like(x)
        # register autograd
        def xpugraph_marker_setup_context(ctx, inputs, output):
            pass
        def xpugraph_marker_backward(ctx, grad_output):
            return grad_output, None  # 对 x 透传梯度, node_id 无梯度
        xpugraph_marker.register_autograd(
            xpugraph_marker_backward,
            setup_context=xpugraph_marker_setup_context,
        )

        # ops.xpugraph.instrument
        @torch.library.custom_op("xpugraph::instrument", mutates_args=())
        def xpugraph_instrument(x: torch.Tensor, node_id: int, model_id: str) -> torch.Tensor:
            align_node = cls.get_node(node_id)
            if model_id not in align_node.data:
                align_node.data[model_id] = []

            align_node.data[model_id].append((
                cls.xorsum32(x), 
                x.clone().detach()
            ))
            return x.clone()
        # register fake
        @xpugraph_instrument.register_fake
        def xpugraph_instrument_fake(x: torch.Tensor, node_id: int, model_id: str) -> torch.Tensor:
            return torch.empty_like(x)
        # register autograd
        def xpugraph_instrument_setup_context(ctx, inputs, output):
            pass
        def xpugraph_instrument_backward(ctx, grad_output):
            return grad_output, None, None  # 对 x 透传梯度, node_id/model_id 无梯度
        xpugraph_instrument.register_autograd(
            xpugraph_instrument_backward,
            setup_context=xpugraph_instrument_setup_context,
        )


    @classmethod
    def get_node(cls, node_id: int) -> AlignmentNode:
        if node_id not in cls._nodes:
            cls._nodes[node_id] = AlignmentNode(node_id)
        return cls._nodes[node_id]


    @staticmethod
    def xorsum32(t: torch.Tensor) -> int:
        b = t.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
        b = F.pad(b, (0, (-b.numel()) % 4))
        w = b.view(-1, 4).to(torch.int64)
        u32 = w[:, 0] | (w[:, 1] << 8) | (w[:, 2] << 16) | (w[:, 3] << 24)
        return int(np.bitwise_xor.reduce(u32.numpy(), dtype=np.uint64)) & 0xFFFFFFFF


def inject_marker_meta_and_remove_marker(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    graph = gm.graph
    markers_to_remove = []

    for node in graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.xpugraph.marker.default:
            # args = (input_tensor, node_id)
            source_node: torch.fx.Node = node.args[0]
            node_id: int = node.args[1]

            # 注入 alignment node_id 到被 mark 的算子的 meta 中
            if "alignment_node_ids" not in source_node.meta:
                source_node.meta["alignment_node_ids"] = []
            source_node.meta["alignment_node_ids"].append(node_id)

            # marker 的输出直接替换为其输入（bypass marker）
            node.replace_all_uses_with(source_node)
            markers_to_remove.append(node)

    for node in markers_to_remove:
        graph.erase_node(node)

    graph.lint()
    gm.recompile()
    return gm


def insert_instrument_nodes(gm: torch.fx.GraphModule, model_id: str) -> torch.fx.GraphModule:
    graph = gm.graph

    for node in list(graph.nodes):
        alignment_ids = node.meta.get("alignment_node_ids", [])
        for node_id in alignment_ids:
            with graph.inserting_after(node):
                instrument_node = graph.call_function(
                    torch.ops.xpugraph.instrument.default,
                    args=(node, node_id, model_id),
                )

    graph.lint()
    gm.recompile()
    return gm


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        out = self.linear(x)
        # 插入 marker: 标记 linear 输出，绑定 alignment node 0
        out = torch.ops.xpugraph.marker(out, 0)
        out = torch.relu(out)
        # 插入 marker: 标记 relu 输出，绑定 alignment node 1
        out = torch.ops.xpugraph.marker(out, 1)
        return out

if __name__ == "__main__":
    
    mgr = AlignmentManager()

    def make_alignment_backend(model_id: str):
        def alignment_fw_compiler(gm: torch.fx.GraphModule, example_inputs):
            print(f"\n[{model_id}] 原始 aot forward graph (含 marker):")
            gm.graph.print_tabular()

            gm = inject_marker_meta_and_remove_marker(gm)
            print(f"\n[{model_id}] 删除 marker 后:")
            gm.graph.print_tabular()

            gm = insert_instrument_nodes(gm, model_id)
            print(f"\n[{model_id}] 插入 instrument 后:")
            gm.graph.print_tabular()

            return gm.forward

        def alignment_bw_compiler(gm: torch.fx.GraphModule, example_inputs):
            return gm.forward

        from torch._dynamo.backends.common import aot_autograd as make_aot_backend
        return make_aot_backend(
            fw_compiler=alignment_fw_compiler,
            bw_compiler=alignment_bw_compiler,
        )

    # ---- 构建两个相同权重的 model，用不同 model_id 编译 ----
    SEED = 42

    # Model A: 原始 float32
    torch.manual_seed(SEED)
    model_a = MyModel()
    compiled_a = torch.compile(model_a, backend=make_alignment_backend("model_A"))

    # Model B: 模拟另一个 device/精度（这里同样 float32，保证 bitwise 对齐）
    torch.manual_seed(SEED)
    model_b = MyModel()
    compiled_b = torch.compile(model_b, backend=make_alignment_backend("model_B"))

    # ---- 同一随机输入，各跑一步 ----
    torch.manual_seed(SEED)
    x = torch.randn(4, 16)

    print("\n" + "=" * 60)
    print(">>> Model A 前向")
    y_a = compiled_a(x.clone())

    print("\n>>> Model B 前向")
    y_b = compiled_b(x.clone())

    # ---- 对比两个 model 在每个 AlignmentNode 上的记录 ----
    print("\n" + "=" * 60)
    print("对齐结果对比:")
    print("=" * 60)

    all_pass = True
    for node_id, align_node in sorted(mgr._nodes.items()):
        data_a = align_node.data.get("model_A", [])
        data_b = align_node.data.get("model_B", [])

        if len(data_a) != len(data_b):
            print(f"\n  [FAIL] AlignmentNode(id={node_id}): 记录数不一致 "
                  f"(model_A={len(data_a)}, model_B={len(data_b)})")
            all_pass = False
            continue

        for step, ((xorsum_a, tensor_a), (xorsum_b, tensor_b)) in enumerate(zip(data_a, data_b)):
            bitwise_match = (xorsum_a == xorsum_b)
            close_match = torch.allclose(tensor_a, tensor_b, atol=1e-6, rtol=1e-5)

            status = "PASS" if bitwise_match else ("CLOSE" if close_match else "FAIL")
            if status != "PASS":
                all_pass = False

            print(f"\n  AlignmentNode(id={node_id}), step={step}:")
            print(f"    model_A  xorsum=0x{xorsum_a:08X}  shape={tensor_a.shape}  dtype={tensor_a.dtype}")
            print(f"    model_B  xorsum=0x{xorsum_b:08X}  shape={tensor_b.shape}  dtype={tensor_b.dtype}")
            print(f"    bitwise={bitwise_match}  allclose={close_match}  => [{status}]")

            if not bitwise_match:
                diff = (tensor_a - tensor_b).abs()
                print(f"    max_diff={diff.max().item():.6e}  mean_diff={diff.mean().item():.6e}")

    print("\n" + "=" * 60)
    print(f"总结: {'ALL PASSED ✓' if all_pass else 'SOME FAILED ✗'}")
    print("=" * 60)