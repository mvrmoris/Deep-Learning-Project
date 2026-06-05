import random
import torch.nn as nn
from typing import Callable
from .network_encoding import (
    OpFactory, Node, Edge, DAG,
    CellSpec, CellNode, CellEdge, NetworkDAG,
)
import re

# ... (le classi sopra) ...

# ── operazioni NB201 ───────────────────────────────────────────────────────────

def op_none() -> OpFactory:
    class Zero(nn.Module):
        def forward(self, x): return x * 0
    def factory(C_in, C_out): return Zero()
    factory.op_name = "none"
    return factory

def op_skip() -> OpFactory:
    def factory(C_in, C_out): return nn.Identity()
    factory.op_name = "skip_connect"
    return factory

def op_conv1x1() -> OpFactory:
    def factory(C_in, C_out):
        return nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out),
        )
    factory.op_name = "nor_conv_1x1"
    return factory

def op_conv3x3() -> OpFactory:
    def factory(C_in, C_out):
        return nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_out),
        )
    factory.op_name = "nor_conv_3x3"
    return factory

def op_avg_pool3x3() -> OpFactory:
    def factory(C_in, C_out): return nn.AvgPool2d(3, stride=1, padding=1)
    factory.op_name = "avg_pool3x3"
    return factory

NB201_OPS: list[OpFactory] = [
    op_none(), op_skip(), op_conv1x1(), op_conv3x3(), op_avg_pool3x3()
]

EDGE_ORDER = [("0","1"), ("0","2"), ("1","2"), ("0","3"), ("1","3"), ("2","3")]


# ── unica funzione ─────────────────────────────────────────────────────────────

def sample_nb201_networks(
    N: int,
    C: int = 16,
    seed: int | None = None,
) -> list[NetworkDAG]:
    """
    Genera N reti NB201 casuali.
    Ogni rete è un NetworkDAG con un solo CellNode,
    che contiene un DAG con 4 nodi e 6 archi campionati.
    """
    rng = random.Random(seed)
    networks = []

    for _ in range(N):
        # campiona 6 op casuali
        ops_6 = [rng.choice(NB201_OPS) for _ in range(6)]

        # costruisce il DAG interno della cella
        dag = DAG(
            nodes=[
                Node("0"),
                Node("1", aggregation="sum"),
                Node("2", aggregation="sum"),
                Node("3", aggregation="sum"),
            ],
            edges=[Edge(s, d, op) for (s, d), op in zip(EDGE_ORDER, ops_6)],
            inputs=["0"],
            outputs=["3"],
        )

        # un solo CellNode, nessun CellEdge
        cell_spec = CellSpec(name="nb201_cell", dag=dag)
        networks.append(NetworkDAG(
            cell_nodes=[CellNode("cell", cell_spec, C_in=C, C_out=C)],
            cell_edges=[],
            inputs=["cell"],
            outputs=["cell"],
        ))

    return networks

_NB201_STR_TO_FACTORY: dict[str, OpFactory] = {
    "none"        : op_none(),
    "skip_connect": op_skip(),
    "nor_conv_1x1": op_conv1x1(),
    "nor_conv_3x3": op_conv3x3(),
    "avg_pool_3x3": op_avg_pool3x3(),
}

_NB201_EDGE_ORDER = [("0","1"), ("0","2"), ("1","2"), ("0","3"), ("1","3"), ("2","3")]

_OP_RE = re.compile(r"([^|~]+)~\d+")

def nasbench201_strings_to_networkdags(arch_strings: list[str]) -> list[NetworkDAG]:
    networks = []
    for arch_str in arch_strings:
        ops = _OP_RE.findall(arch_str)
        if len(ops) != 6:
            raise ValueError(f"Attese 6 op, trovate {len(ops)}: {arch_str!r}")

        dag = DAG(
            nodes=[
                Node("0"),
                Node("1", aggregation="sum"),
                Node("2", aggregation="sum"),
                Node("3", aggregation="sum"),
            ],
            edges=[
                Edge(s, d, _NB201_STR_TO_FACTORY[op])
                for (s, d), op in zip(_NB201_EDGE_ORDER, ops)
            ],
            inputs=["0"],
            outputs=["3"],
        )
        cell_spec = CellSpec(name=arch_str, dag=dag)
        networks.append(NetworkDAG(
            cell_nodes=[CellNode("cell", cell_spec, C_in=16, C_out=16)],
            cell_edges=[],
            inputs=["cell"],
            outputs=["cell"],
        ))
    return networks