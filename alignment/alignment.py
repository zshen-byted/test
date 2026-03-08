import torch
import torch.nn as nn
import torch.nn.functional as F
import functorch

import torch._dynamo as dynamo
from torch._functorch.aot_autograd import aot_module_simplified
from torch._dynamo.backends.common import aot_autograd

from torch.utils._python_dispatch import TorchDispatchMode
from torch.overrides import TorchFunctionMode

import numpy as np
import graphviz
from typing import Optional

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.llama3 import Llama3


# dynamo.config.error_on_recompile = True

class AlignmentNode:
    def __init__(self, id: int, stage: str):
        self.id = id
        self.stage = stage  # "forward" | "backward"
        self.meta = {} # {"name": "xxx", "nn_module_stack": [...], ...}
        self.data = {} # {model_id: [(xorsum, raw_tensor), ...]}


class AlignmentManager:
    _instance: Optional["AlignmentManager"] = None
    _nodes: dict[int, AlignmentNode] = {}
    _topology: dict[int, set[int]] = {}  # node_id → {successor_node_ids}
    _grad_links: dict[int, int] = {}     # fw_node_id → bw_node_id

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
            ctx.fw_node_id = inputs[1]  # save node_id for backward marker injection
        def xpugraph_marker_backward(ctx, grad_output):
            # inject a backward marker so bw_compiler can instrument gradients
            marked_grad = torch.ops.xpugraph.marker(grad_output, ctx.fw_node_id)
            return marked_grad, None
        xpugraph_marker.register_autograd(
            xpugraph_marker_backward,
            setup_context=xpugraph_marker_setup_context,
        )

        # ops.xpugraph.instrument
        @torch.library.custom_op("xpugraph::instrument", mutates_args=())
        def xpugraph_instrument(x: torch.Tensor, node_id: int, model_id: str, stage: str) -> torch.Tensor:
            align_node = cls.get_node(node_id, stage=stage)
            if model_id not in align_node.data:
                align_node.data[model_id] = []

            align_node.data[model_id].append((
                cls.xorsum32(x), 
                x.clone().detach()
            ))
            return torch.empty(0)
        # register fake
        @xpugraph_instrument.register_fake
        def xpugraph_instrument_fake(x: torch.Tensor, node_id: int, model_id: str, stage: str) -> torch.Tensor:
            return torch.empty(0)
        # register autograd
        def xpugraph_instrument_setup_context(ctx, inputs, output):
            pass
        def xpugraph_instrument_backward(ctx, grad_output):
            return None, None, None, None  # x, node_id, model_id, stage
        xpugraph_instrument.register_autograd(
            xpugraph_instrument_backward,
            setup_context=xpugraph_instrument_setup_context,
        )

    @classmethod
    def get_node(cls, node_id: int, stage: str = "forward") -> AlignmentNode:
        if node_id not in cls._nodes:
            cls._nodes[node_id] = AlignmentNode(node_id, stage)
        return cls._nodes[node_id]

    @classmethod
    def add_edge(cls, src_id: int, dst_id: int):
        if src_id not in cls._topology:
            cls._topology[src_id] = set()
        cls._topology[src_id].add(dst_id)

    @classmethod
    def add_grad_link(cls, fw_node_id: int, bw_node_id: int):
        cls._grad_links[fw_node_id] = bw_node_id

    @classmethod
    def print_data_comparison(cls, model_ids: list[str], gold_model_id: Optional[str] = None, stage: str = "forward"):
        gold_model_id = gold_model_id if gold_model_id is not None else model_ids[0]
        # filter nodes by stage
        stage_nodes = {nid: n for nid, n in cls._nodes.items() if n.stage == stage}
        if not stage_nodes:
            print(f"No {stage} alignment nodes found.")
            return
        first_node_data = list(stage_nodes.values())[0].data.get(gold_model_id, [])
        n_steps = len(first_node_data)
        for i in range(n_steps):
            print(f"\n{'='*40}\nStep {i} ({stage})\n{'='*40}")
            for node_id, align_node in sorted(stage_nodes.items()):
                try:
                    print(f"AlignmentNode {node_id}:")
                    gold_data = align_node.data.get(gold_model_id, [])
                    if not gold_data:
                        print(f"  Gold model {gold_model_id} has no data for this node.")
                        continue
                    for model_id in model_ids:
                        data = align_node.data.get(model_id, [])
                        if not data:
                            print(f"  {model_id}: No data collected")
                            continue
                        print(f"  {model_id+(' (gold)' if model_id == gold_model_id else ''):15}:", end="")
                        (xorsum, raw_tensor) = data[i]
                        gold_xorsum, gold_tensor = gold_data[i]
                        raw_tensor_32, gold_tensor_32 = raw_tensor.to(torch.float32), gold_tensor.to(torch.float32)

                        max_diff:float = (raw_tensor_32 - gold_tensor_32).abs().max().item()
                        closeto:bool = torch.allclose(raw_tensor_32, gold_tensor_32, rtol=1e-3, atol=1e-5)

                        print(f"dtype={raw_tensor.dtype}, xorsum=0x{xorsum:08X}, max_diff={max_diff:.8f}, closeto={str(closeto):5}")
                    print("")
                except Exception as e:
                    print("\n")
                    continue

    @staticmethod
    def xorsum32(t: torch.Tensor) -> int:
        b = t.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
        b = F.pad(b, (0, (-b.numel()) % 4))
        w = b.view(-1, 4).to(torch.int64)
        u32 = w[:, 0] | (w[:, 1] << 8) | (w[:, 2] << 16) | (w[:, 3] << 24)
        return int(np.bitwise_xor.reduce(u32.numpy(), dtype=np.uint64)) & 0xFFFFFFFF

    @classmethod
    def export_dot(
        cls,
        model_ids: list[str],
        gold_model_id: Optional[str] = None,
        step: int = 0,
        output_path: str = "dependency_graph",
    ) -> graphviz.Digraph:
        """Export alignment topology + data comparison as a Graphviz DOT file."""
        import json as _json

        gold_model_id = gold_model_id if gold_model_id is not None else model_ids[0]

        fw_nodes = {nid: n for nid, n in cls._nodes.items() if n.stage == "forward"}
        bw_nodes = {nid: n for nid, n in cls._nodes.items() if n.stage == "backward"}

        def _build_node_attrs(node_id: int, align_node: AlignmentNode) -> dict[str, str]:
            """Return graphviz node attributes: visible label + hidden data."""
            op_name = align_node.meta.get("op_name", "?")
            label = f"{node_id}: {op_name}"

            # build per-model detail dict for tooltip / comment
            gold_data = align_node.data.get(gold_model_id, [])
            details: list[dict] = []
            for mid in model_ids:
                entry: dict = {"model": mid}
                data = align_node.data.get(mid, [])
                if not data or step >= len(data):
                    entry["status"] = "no data"
                else:
                    xorsum, raw_tensor = data[step]
                    entry["dtype"] = str(raw_tensor.dtype).replace("torch.", "")
                    entry["xorsum"] = f"0x{xorsum:08X}"
                    if gold_data and step < len(gold_data):
                        g32 = gold_data[step][1].to(torch.float32)
                        r32 = raw_tensor.to(torch.float32)
                        entry["max_diff"] = float(f"{(r32 - g32).abs().max().item():.8f}")
                        entry["closeto"] = bool(torch.allclose(r32, g32, rtol=1e-3, atol=1e-5))
                details.append(entry)

            # tooltip: human-readable multi-line summary (shown on hover)
            nbsp = u"\u00A0"
            tip_lines = [f"{'-'*68}\nNode {node_id} ({align_node.stage}) — {op_name}\n{'-'*68}"]
            for d in details:
                if d.get("status") == "no data":
                    tip_lines.append(f"  {d['model']}: no data")
                else:
                    flag = '✓' if d.get('closeto', False) else '✗'
                    gold_sign = '(gold)' if d['model'] == gold_model_id else ' '
                    tip_lines.append(
                        f"{d['model']+':':10} {d.get('dtype','?'):8} "
                        f"xor={d.get('xorsum','?')} Δ={d.get('max_diff','?'):.9f} {flag} {gold_sign}"
                    )
            tooltip = "\n".join(tip_lines).replace(" ", nbsp)

            # comment: machine-readable JSON (parseable by custom tools)
            comment = _json.dumps({"node_id": node_id, "stage": align_node.stage,
                                   "op_name": op_name, "models": details}, ensure_ascii=False)

            color = "#4472C4" if align_node.stage == "forward" else "#AD683A"
            return {
                "label": label,
                "tooltip": tooltip,
                "comment": comment,
                "style": "filled",
                "fillcolor": "#DEEBF7" if align_node.stage == "forward" else "#FBE5D6",
                "color": color,
                "fontcolor": "#333333",
            }

        g = graphviz.Digraph(
            "AlignmentGraph",
            graph_attr={"rankdir": "TB", "newrank": "true",
                        "fontname": "Helvetica", "fontsize": "12"},
            node_attr={"shape": "box", "style": "rounded,filled", "fontname": "Courier", "fontsize": "10"},
            edge_attr={"fontsize": "9"},
        )

        # forward cluster (will appear on the left)
        with g.subgraph(name="cluster_forward") as fw:
            fw.attr(label="Forward", style="solid", color="#4472C4",
                    fontcolor="#4472C4", fontsize="18")
            for nid in sorted(fw_nodes):
                fw.node(f"n{nid}", **_build_node_attrs(nid, fw_nodes[nid]))

        # backward cluster (will appear on the right)
        with g.subgraph(name="cluster_backward") as bw:
            bw.attr(label="Backward", style="solid", color="#ED7D31",
                    fontcolor="#ED7D31", fontsize="18")
            for nid in sorted(bw_nodes):
                bw.node(f"n{nid}", **_build_node_attrs(nid, bw_nodes[nid]))

        # rank=same constraints: force each fw/bw pair to the same horizontal row
        for fw_id, bw_id in sorted(cls._grad_links.items()):
            with g.subgraph() as s:
                s.attr(rank="same")
                s.node(f"n{fw_id}")
                s.node(f"n{bw_id}")

        # topology edges
        for src_id, dst_ids in sorted(cls._topology.items()):
            for dst_id in sorted(dst_ids):
                g.edge(f"n{src_id}", f"n{dst_id}")

        # grad_link edges (forward → backward)
        for fw_id, bw_id in sorted(cls._grad_links.items()):
            g.edge(f"n{fw_id}", f"n{bw_id}", style="dotted", color="#000000", constraint="false")

        g.save(f"{output_path}.dot")
        print(f"DOT file written to {output_path}.dot")
        return g

    @classmethod
    def inject_marker_meta_and_remove_marker_fw_pass(cls, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Forward pass: inject alignment node metadata and remove marker ops."""
        graph = gm.graph
        markers_to_remove = []

        for node in graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.xpugraph.marker.default:
                # args = (input_tensor, node_id)
                source_node: torch.fx.Node = node.args[0]
                node_id: int = node.args[1]
                align_node = cls.get_node(node_id, stage="forward")

                # save the instrumented op's name into meta
                op_name = str(source_node.target) if source_node.op == "call_function" else source_node.name
                align_node.meta["op_name"] = op_name

                if "align_node_ids" not in source_node.meta:
                    source_node.meta["align_node_ids"] = []
                source_node.meta["align_node_ids"].append(node_id)

                node.replace_all_uses_with(source_node)
                markers_to_remove.append(node)

        for node in markers_to_remove:
            graph.erase_node(node)

        graph.lint()
        gm.recompile()
        return gm

    @classmethod
    def inject_marker_meta_and_remove_marker_bw_pass(cls, gm: torch.fx.GraphModule, bw_node_id_offset: int = 0) -> torch.fx.GraphModule:
        """Backward pass: inject alignment node metadata with offset, register grad links, and remove marker ops."""
        graph = gm.graph
        markers_to_remove = []

        for node in graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.xpugraph.marker.default:
                # args = (input_tensor, node_id)
                source_node: torch.fx.Node = node.args[0]
                fw_node_id: int = node.args[1]

                node_id = fw_node_id + bw_node_id_offset
                cls.add_grad_link(fw_node_id, node_id)

                align_node = cls.get_node(node_id, stage="backward")

                # save the instrumented op's name into meta
                op_name = str(source_node.target) if source_node.op == "call_function" else source_node.name
                align_node.meta["op_name"] = op_name

                if "align_node_ids" not in source_node.meta:
                    source_node.meta["align_node_ids"] = []
                source_node.meta["align_node_ids"].append(node_id)

                node.replace_all_uses_with(source_node)
                markers_to_remove.append(node)

        for node in markers_to_remove:
            graph.erase_node(node)

        graph.lint()
        gm.recompile()
        return gm

    @staticmethod
    def insert_instrument_nodes_pass(gm: torch.fx.GraphModule, model_id: str, stage: str = "forward") -> torch.fx.GraphModule:
        graph = gm.graph

        for node in list(graph.nodes):
            align_node_ids = node.meta.get("align_node_ids", [])
            for node_id in align_node_ids:
                with graph.inserting_after(node):
                    instrument_node = graph.call_function(
                        torch.ops.xpugraph.instrument.default,
                        args=(node, node_id, model_id, stage),
                    )

        graph.lint()
        gm.recompile()
        return gm

    @classmethod
    def build_topology_from_graph(cls, graph: torch.fx.Graph):
        """topology builder using dataflow coloring.
        For each FX node, tracks which align_node_ids' data flows through it.
        When encountering a node with align_node_ids, builds edges from
        the incoming (upstream) align_node_ids, then resets flow to own ids."""
        reaching: dict[torch.fx.Node, set[int]] = {}

        for node in graph.nodes:  # FX graph guarantees topological order
            incoming: set[int] = set()
            for inp in node.all_input_nodes:
                incoming |= reaching.get(inp, set())
            own_ids = set(node.meta.get("align_node_ids", []))
            if own_ids:
                # build edges: upstream align nodes → this align node
                for s in incoming:
                    for d in own_ids:
                        if s != d:
                            cls.add_edge(s, d)
                # reset: downstream sees only this node's ids
                reaching[node] = own_ids
            else:
                # passthrough: propagate upstream ids
                reaching[node] = incoming


class AlignedModelGenerator:

    class MarkerInjectFuntionMode(TorchFunctionMode):
    
        def __init__(self, ctx: "AlignedModelGenerator"):
            super().__init__()
            self._ctx = ctx
            self._disabled: bool = False

        def __torch_function__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if not self._disabled:
                result = self._inject_markers(result)
            return result

        def _inject_markers(self, result):
            """Recursively inject markers on tensor outputs, handling multi-output ops."""
            if isinstance(result, torch.Tensor):
                if result.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                    return torch.ops.xpugraph.marker(result, self._ctx._new_node_id())
            elif isinstance(result, (tuple, list)) and all(isinstance(item, torch.Tensor) for item in result):
                items = [self._inject_markers(item) for item in result]
                return type(result)(items)   # handles tuple, list
            return result
    
    class EagerInstrumentFunctionMode(TorchFunctionMode):
        
        def __init__(self, ctx: "AlignedModelGenerator", model_id: str):
            super().__init__()
            self._ctx = ctx
            self._model_id = model_id
            self._next_node_id = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            self._instrument(result)
            return result

        def _instrument(self, result):
            """Recursively instrument tensor outputs and register backward hooks, handling multi-output ops."""
            if isinstance(result, torch.Tensor):
                if result.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                    node_id = self._next_node_id
                    self._next_node_id += 1
                    # call instrument op directly to collect data in eager mode
                    torch.ops.xpugraph.instrument(result, node_id, self._model_id, "forward")
                    if result.requires_grad:
                        def _bw_hook(grad, nid=node_id, mode=self):
                            bw_nid = nid + mode._next_node_id # at backward time, mode._next_node_id = N (total fw nodes)
                            AlignmentManager.add_grad_link(nid, bw_nid)
                            torch.ops.xpugraph.instrument(grad, bw_nid, mode._model_id, "backward")
                        result.register_hook(_bw_hook)
            elif isinstance(result, (tuple, list)) and all(isinstance(item, torch.Tensor) for item in result):
                for item in result:
                    self._instrument(item)

    def __init__(self, model_id: str):
        self.align_mgr = AlignmentManager()
        self.model_id = model_id
        self._next_node_id = 0
        self._marker_inject_mode = self.MarkerInjectFuntionMode(self)
        self._eager_instrument_mode = self.EagerInstrumentFunctionMode(self, model_id)
        self._generated = False # one AlignedModelGenerator instance could only generate one compiled/eager model.

    def _new_node_id(self):
        ret = self._next_node_id
        self._next_node_id += 1
        return ret

    def _make_backend(self):

        fw_marker_count = [0]  # share between fw and bw compilers

        def alignment_fw_compiler(gm: torch.fx.GraphModule, example_inputs):
            # count forward markers before removing them (for backward ID offset)
            fw_marker_count[0] = sum(
                1 for n in gm.graph.nodes
                if n.op == "call_function" and n.target == torch.ops.xpugraph.marker.default
            )
            gm = self.align_mgr.inject_marker_meta_and_remove_marker_fw_pass(gm)
            self.align_mgr.build_topology_from_graph(gm.graph)
            gm = self.align_mgr.insert_instrument_nodes_pass(gm, self.model_id, stage="forward")

            print("\n\n=== Forward Instrumented Graph ===", flush=True)
            gm.graph.print_tabular()

            return functorch.compile.make_boxed_func(gm.forward)

        def alignment_bw_compiler(gm: torch.fx.GraphModule, example_inputs):
            bw_offset = fw_marker_count[0]  # total forward marker count
            gm = self.align_mgr.inject_marker_meta_and_remove_marker_bw_pass(gm, bw_node_id_offset=bw_offset)
            self.align_mgr.build_topology_from_graph(gm.graph)
            gm = self.align_mgr.insert_instrument_nodes_pass(gm, self.model_id, stage="backward")

            print("\n\n=== Backward Instrumented Graph ===", flush=True)
            gm.graph.print_tabular()

            return functorch.compile.make_boxed_func(gm.forward)

        def backend(gm: torch.fx.GraphModule, example_inputs):
            gm = aot_module_simplified(
                gm,
                example_inputs,
                fw_compiler=alignment_fw_compiler,
                bw_compiler=alignment_bw_compiler,
            )

            tracing_context = torch._guards.TracingContext.try_get()
            orig_guards: torch._guards.GuardsSet = tracing_context.guards_context.dynamo_guards
            filter_flags = [False for _ in orig_guards]
            filtered_guards = torch._guards.GuardsSet(
                set(guard for guard, flag in zip(orig_guards, filter_flags) if not flag)
            )
            tracing_context.guards_context.dynamo_guards = orig_guards - filtered_guards

            return gm

        return backend
    
    def get_compiled(self, model: nn.Module, example_inputs) -> nn.Module:
        if self._generated:
            raise RuntimeError("A AlignedModelGenerator instance can only generate once. Please create a new instance for another model.")

        with self._marker_inject_mode:
            compiled_fn = torch.compile(model, backend=self._make_backend(), dynamic=True, fullgraph=True)
            _ = compiled_fn(*example_inputs) # warmup to trigger compilation and data collection
            _.backward(torch.ones_like(_))
        self._marker_inject_mode._disabled = True
        
        for node in self.align_mgr._nodes.values(): # clear manager node data during warmup run
            node.data.clear()

        class _WrapModuleCompile(nn.Module):
            def __init__(wself, fn):
                super().__init__()
                wself.fn = fn

            @dynamo.disable
            def forward(wself, *args, **kwargs):
                with self._marker_inject_mode: # this 'with' is for hacking dynamo torch_function_stack checker guard.
                    return wself.fn(*args, **kwargs)
                
        self._generated = True
        return _WrapModuleCompile(compiled_fn)

    def get_eager(self, model: nn.Module) -> nn.Module:
        if self._generated:
            raise RuntimeError("A AlignedModelGenerator instance can only generate once. Please create a new instance for another model.")
        class _WrapModule(nn.Module):
            def __init__(wself, model):
                super().__init__()
                wself._model = model

            def forward(wself, *args, **kwargs):
                self._eager_instrument_mode._next_node_id = 0 # reset node_id counter before each forward
                with self._eager_instrument_mode:
                    return wself._model(*args, **kwargs)

        self._generated = True
        return _WrapModule(model)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        out = self.linear(x)
        out = torch.relu(out)
        return out


if __name__ == "__main__":

    torch.manual_seed(42)
    llama_config = {
        "vocab_size": 64,
        "dim": 128,
        "n_layers": 1,
        "n_heads": 4,
        "n_kv_heads": 4,
    }
    
    mgr = AlignmentManager()

    with torch.random.fork_rng():
        eager_32 = AlignedModelGenerator("eager_32").get_eager(Llama3(**llama_config))

    with torch.random.fork_rng():
        comp_32 = AlignedModelGenerator("comp_32").get_compiled(Llama3(**llama_config), (torch.randint(0, 64, (4, 16)),))

    with torch.random.fork_rng():
        eager_16 = AlignedModelGenerator("eager_16").get_eager(Llama3(**llama_config).half())
    
    with torch.random.fork_rng():
        comp_16 = AlignedModelGenerator("comp_16").get_compiled(Llama3(**llama_config).half(), (torch.randint(0, 64, (4, 16)),))


    for i in range(2):
        x = torch.randint(0, 64, (4, 16))

        _ = eager_32(x.clone())
        _.backward(torch.ones_like(_))

        _ = comp_32(x.clone())
        _.backward(torch.ones_like(_))

        _ = eager_16(x.clone())
        _.backward(torch.ones_like(_))

        _ = comp_16(x.clone())
        _.backward(torch.ones_like(_))


    mgr.print_data_comparison(["eager_32", "comp_32", "eager_16", "comp_16"], gold_model_id="eager_32", stage="forward")
    mgr.print_data_comparison(["eager_32", "comp_32", "eager_16", "comp_16"], gold_model_id="eager_32", stage="backward")

    mgr.export_dot(
        ["eager_32", "comp_32", "eager_16", "comp_16"],
        gold_model_id="eager_32",
        step=0,
        output_path="deps_graph",
    )