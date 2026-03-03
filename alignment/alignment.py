import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.library import Library

import torch._dynamo as dynamo
from torch._functorch.aot_autograd import aot_module_simplified
from torch._dynamo.backends.common import aot_autograd

from torch.utils._python_dispatch import TorchDispatchMode
from torch.overrides import TorchFunctionMode
from torch._subclasses.fake_tensor import FakeTensorMode

import numpy as np
import collections
from typing import Optional, Tuple, List, NamedTuple


dynamo.config.error_on_recompile = True

class AlignmentNode:
    def __init__(self, id:int):
        self.id = id
        self.meta = {} # {"name": "xxx", "stage": "forward"/"backward", "nn_module_stack": [...], ...}
        self.data = {} # {"model_1": [(xorsum, raw_tensor, ...), ...], "model_2": [(xorsum, raw_tensor, ...), ...], ...}


class AlignmentManager:
    _instance: Optional["AlignmentManager"] = None
    _nodes: dict[int, dict] = {}

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
            return x.clone()
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
            return torch.empty(0)
        # register fake
        @xpugraph_instrument.register_fake
        def xpugraph_instrument_fake(x: torch.Tensor, node_id: int, model_id: str) -> torch.Tensor:
            return torch.empty(0)
        # register autograd
        def xpugraph_instrument_setup_context(ctx, inputs, output):
            pass
        def xpugraph_instrument_backward(ctx, grad_output):
            return None, None, None
        xpugraph_instrument.register_autograd(
            xpugraph_instrument_backward,
            setup_context=xpugraph_instrument_setup_context,
        )

    @classmethod
    def get_node(cls, node_id: int) -> AlignmentNode:
        if node_id not in cls._nodes:
            cls._nodes[node_id] = AlignmentNode(node_id)
        return cls._nodes[node_id]

    @classmethod
    def print_data_comparison(cls, model_ids: list[str], gold_model_id: Optional[str] = None):
        gold_model_id = gold_model_id if gold_model_id is not None else model_ids[0]
        for node_id, align_node in sorted(cls._nodes.items()):
            print(f"\nAlignmentNode {node_id}:")
            gold_data = align_node.data.get(gold_model_id, [])
            if not gold_data:
                print(f"  Gold model {gold_model_id} has no data collected for this node.")
                continue
            for model_id in model_ids:
                data = align_node.data.get(model_id, [])
                if not data:
                    print(f"  {model_id}: No data collected")
                    continue
                print(f"  {model_id}{' (gold)' if model_id == gold_model_id else '':8}:", end="")
                for i, (xorsum, raw_tensor) in enumerate(data):
                    gold_xorsum, gold_tensor = gold_data[i]
                    raw_tensor_32, gold_tensor_32 = raw_tensor.to(torch.float32), gold_tensor.to(torch.float32)

                    max_diff:float = (raw_tensor_32 - gold_tensor_32).abs().max().item()
                    closeto:bool = torch.allclose(raw_tensor_32, gold_tensor_32, rtol=1e-3, atol=1e-5)

                    print(f"dtype={raw_tensor.dtype}, xorsum=0x{xorsum:08X}, max_diff={max_diff:.8f}, closeto={str(closeto):5}")

    @staticmethod
    def xorsum32(t: torch.Tensor) -> int:
        b = t.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
        b = F.pad(b, (0, (-b.numel()) % 4))
        w = b.view(-1, 4).to(torch.int64)
        u32 = w[:, 0] | (w[:, 1] << 8) | (w[:, 2] << 16) | (w[:, 3] << 24)
        return int(np.bitwise_xor.reduce(u32.numpy(), dtype=np.uint64)) & 0xFFFFFFFF



def inject_marker_meta_and_remove_marker_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    graph = gm.graph
    markers_to_remove = []

    for node in graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.xpugraph.marker.default:
            # args = (input_tensor, node_id)
            source_node: torch.fx.Node = node.args[0]
            node_id: int = node.args[1]

            # 注入 alignment node_id 到被 mark 的算子的 meta 中
            if "align_node_ids" not in source_node.meta:
                source_node.meta["align_node_ids"] = []
            source_node.meta["align_node_ids"].append(node_id)

            # marker 的输出直接替换为其输入（bypass marker）
            node.replace_all_uses_with(source_node)
            markers_to_remove.append(node)

    for node in markers_to_remove:
        graph.erase_node(node)

    graph.lint()
    gm.recompile()
    return gm


def insert_instrument_nodes_pass(gm: torch.fx.GraphModule, model_id: str) -> torch.fx.GraphModule:
    graph = gm.graph

    for node in list(graph.nodes):
        align_node_ids = node.meta.get("align_node_ids", [])
        for node_id in align_node_ids:
            with graph.inserting_after(node):
                instrument_node = graph.call_function(
                    torch.ops.xpugraph.instrument.default,
                    args=(node, node_id, model_id),
                )

    graph.lint()
    gm.recompile()
    return gm

class AlignmentManagerContext(TorchFunctionMode):
    def __init__(self, model_id: str, function_inject=False, module_hook_inject=False, module:Optional[nn.Module]=None):
        super().__init__()
        self.model_id = model_id
        self.align_mgr = AlignmentManager()
        self._next_node_id = 0
        self.function_inject = function_inject
        #TODO module_hook_inject
    
    def __torch_function__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if self._should_inject_function_marker(func, types, result):
            result = torch.ops.xpugraph.marker(result, self._new_node_id())
        return result

    def _should_inject_function_marker(self, func, types, result) -> bool:
        ret = self.function_inject
        if not isinstance(result, torch.Tensor):
            ret = False

        return ret

    def _new_node_id(self):
        ret = self._next_node_id
        self._next_node_id += 1
        return ret

    def get_backend(self):
        def alignment_fw_compiler(gm: torch.fx.GraphModule, example_inputs):
            gm.graph.print_tabular()
            
            print("", flush=True)
            print("\n=== Injecting marker meta and removing marker nodes ===", flush=True)
            gm = inject_marker_meta_and_remove_marker_pass(gm)
            gm.graph.print_tabular()
            
            print("", flush=True)
            print("\n=== Inserting instrument nodes ===", flush=True)
            gm = insert_instrument_nodes_pass(gm, self.model_id)
            gm.graph.print_tabular()
            print("", flush=True)


            return gm.forward

        def alignment_bw_compiler(gm: torch.fx.GraphModule, example_inputs):
            return gm.forward

        def backend(gm: torch.fx.GraphModule, example_inputs):

            gm = aot_module_simplified(
                gm,
                example_inputs,
                fw_compiler=alignment_fw_compiler,
                bw_compiler=alignment_bw_compiler,
            )

            tracing_context = torch._guards.TracingContext.try_get()
            orig_guards:torch._guards.GuardsSet= tracing_context.guards_context.dynamo_guards
            filter_flags = [False for _ in orig_guards]
            filtered_guards = torch._guards.GuardsSet(
                set(guard for guard, flag in zip(orig_guards, filter_flags) if not flag)
            )
            tracing_context.guards_context.dynamo_guards = orig_guards - filtered_guards

            return gm
        return backend


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        out = self.linear(x)
        out = torch.relu(out)
        return out


if __name__ == "__main__":

    SEED = 42
    torch.manual_seed(SEED)
    
    mgr = AlignmentManager()

    # Model A: 原始 float32
    with torch.random.fork_rng():
        model_a = MyModel()
        input = torch.randn(4, 16)
        with AlignmentManagerContext("model_A", function_inject=True) as am_ctx:
            compiled_a = torch.compile(model_a, backend=am_ctx.get_backend(), dynamic=True, fullgraph=True, )
            _ = compiled_a(input)


    print("\n=== Running inference to collect data ===", flush=True)

    x = torch.randn(4, 16)

    y_a = compiled_a(x.clone())


    # # Model B: 同样的模型
    # with torch.random.fork_rng():
    #     model_b = MyModel()
    #     input = torch.randn(4, 16)
    #     with AlignmentManagerContext("model_B", function_inject=True) as am_ctx:
    #         compiled_b = torch.compile(model_b, backend=am_ctx.get_backend(), dynamic=True, fullgraph=True)
    #         _ = compiled_b(input)

    # # Model C: 量化到float16
    # with torch.random.fork_rng():
    #     model_c = MyModel().half()
    #     input = torch.randn(4, 16).half()
    #     with AlignmentManagerContext("model_C", function_inject=True) as am_ctx:
    #         compiled_c = torch.compile(model_c, backend=am_ctx.get_backend(), dynamic=True, fullgraph=True)
    #         _ = compiled_c(input)

    print("\n=== Running inference to collect data ===", flush=True)

    x = torch.randn(4, 16)

    y_a = compiled_a(x.clone())

    # y_b = compiled_b(x.clone())

    # y_c = compiled_c(x.clone().half())


    # # ----  对比所有model 在每个 AlignmentNode 上的记录 ----
    # print(f"\n{'='*60}\n对齐结果对比:\n{'='*60}")

    # mgr.print_data_comparison(["model_A", "model_B", "model_C"])
