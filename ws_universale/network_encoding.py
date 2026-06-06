import random
import torch.nn as nn
from typing import Callable
OpFactory = Callable[[int, int], nn.Module]

OpFactory = Callable[[int, int], nn.Module]

# ── struttura dati ─────────────────────────────────────────────────────────────

class Node:
    def __init__(self, id: str, aggregation: str = "sum"):
        self.id = id
        self.aggregation = aggregation

    def __repr__(self):
        return f"Node({self.id!r}, agg={self.aggregation!r})"

class Edge:
    def __init__(self, src: str, dst: str, op: OpFactory):
        self.src = src
        self.dst = dst
        self.op  = op

    def __repr__(self):
        return f"Edge({self.src!r}→{self.dst!r}, op={self.op.__name__ if hasattr(self.op,'__name__') else self.op})"

class DAG:
    def __init__(self, nodes, edges, inputs, outputs):
        self.nodes   = nodes
        self.edges   = edges
        self.inputs  = inputs
        self.outputs = outputs
        self._in_edges  = {n.id: [] for n in nodes} 
        self._out_edges = {n.id: [] for n in nodes}
        for e in edges:
            self._in_edges[e.dst].append(e)
            self._out_edges[e.src].append(e)

    def in_edges(self, node_id): return self._in_edges[node_id]
    def topological_order(self):
        deg = {n.id: len(self._in_edges[n.id]) for n in self.nodes}
        q = [nid for nid, d in deg.items() if d == 0]
        out = []
        while q:
            nid = q.pop(0); out.append(nid)
            for e in self._out_edges[nid]:
                deg[e.dst] -= 1
                if deg[e.dst] == 0: q.append(e.dst)
        return out

class CellSpec:
    def __init__(self, name: str, dag: DAG):
        self.name = name
        self.dag  = dag

    def __repr__(self): return f"CellSpec({self.name!r})"

class CellNode:
    def __init__(self, id: str, cell_spec: CellSpec, C_in: int, C_out: int):
        self.id        = id
        self.cell_spec = cell_spec
        self.C_in      = C_in
        self.C_out     = C_out

    def __repr__(self):
        return f"CellNode({self.id!r}, {self.cell_spec.name!r}, {self.C_in}→{self.C_out})"

class CellEdge:
    def __init__(self, src: str, dst: str):
        self.src = src
        self.dst = dst

class NetworkDAG:
    def __init__(self, cell_nodes, cell_edges, inputs, outputs):
        self.cell_nodes = cell_nodes
        self.cell_edges = cell_edges
        self.inputs     = inputs
        self.outputs    = outputs

    def __repr__(self):
        return f"NetworkDAG({len(self.cell_nodes)} celle)"

