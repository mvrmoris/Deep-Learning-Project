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

# ── mappa op_name → stringa NB201 ─────────────────────────────────────────────
# Le stringhe ufficiali NB201/NATS-Bench TSS
_OP_TO_NB201 = {
    "none"        : "none",
    "skip_connect": "skip_connect",
    "nor_conv_1x1": "nor_conv_1x1",
    "nor_conv_3x3": "nor_conv_3x3",
    "avg_pool3x3" : "avg_pool_3x3",
    "avg_pool_3x3": "avg_pool_3x3"  # Inserito per doppia sicurezza
}

_EDGE_ORDER = [("0","1"), ("0","2"), ("0","3"), ("1","2"), ("1","3"), ("2","3")]


def networkdag_to_nb201_str(net) -> str:
    """
    Converte un NetworkDAG NB201 nella stringa architetturale ufficiale.
    Formato: |op~0|+|op~0|op~1|+|op~0|op~1|op~2|
    dove ogni gruppo corrisponde agli archi entranti in un nodo.

    nodo 1 riceve da: (0→1)
    nodo 2 riceve da: (0→2), (1→2)
    nodo 3 riceve da: (0→3), (1→3), (2→3)
    """
    dag = net.cell_nodes[0].cell_spec.dag

    # costruisce mappa (src,dst) → op_name
    edge_map = {}
    for edge in dag.edges:
        edge_map[(edge.src, edge.dst)] = getattr(edge.op, "op_name", None)

    def nb201_op(src, dst):
        op_name = edge_map.get((src, dst))
        assert op_name is not None, f"arco ({src},{dst}) non trovato"
        nb201 = _OP_TO_NB201.get(op_name)
        assert nb201 is not None, f"op_name '{op_name}' non ha mapping NB201"
        return nb201

    # gruppo nodo 1: arco (0→1)
    g1 = f"|{nb201_op('0','1')}~0|"
    # gruppo nodo 2: archi (0→2), (1→2)
    g2 = f"|{nb201_op('0','2')}~0|{nb201_op('1','2')}~1|"
    # gruppo nodo 3: archi (0→3), (1→3), (2→3)
    g3 = f"|{nb201_op('0','3')}~0|{nb201_op('1','3')}~1|{nb201_op('2','3')}~2|"

    return f"{g1}+{g2}+{g3}"