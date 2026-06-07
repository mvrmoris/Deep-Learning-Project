from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn

from .network_encoding import (
    Node, DAG,
    CellSpec, CellNode, CellEdge, NetworkDAG,
)

# ── Tipo interno per identificare un percorso ────────────────────────────────
# frozenset[tuple[src, dst, op_name]]
ArchPath = frozenset


# ── Stem e Head ──────────────────────────────────────────────────────────────

class Stem(nn.Module):
    """Conv3x3 + BN + ReLU: porta da 3 canali a C canali."""
    def __init__(self, C_out: int = 16):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(3, C_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Head(nn.Module):
    """Global Average Pooling + Linear classifier."""
    def __init__(self, C_in: int = 16, n_classes: int = 10):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.gap(x).flatten(1))


# ── MixedOp ──────────────────────────────────────────────────────────────────

class MixedOp(nn.Module):
    """
    Contenitore di operazioni su un arco della supernet.
    Ogni op è istanziata una sola volta — weight sharing garantito.
    """
    def __init__(self, ops: list):
        super().__init__()
        self.ops = nn.ModuleList(ops)

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return self.ops[idx](x)

    def op_names(self) -> list:
        return [getattr(op, "op_name", type(op).__name__) for op in self.ops]

    def __repr__(self) -> str:
        return f"MixedOp({self.op_names()})"


# ── SuperEdge ────────────────────────────────────────────────────────────────

class SuperEdge:
    """Arco nella supernet con le sue operazioni candidate."""
    def __init__(self, src: str, dst: str, mixed_op: MixedOp,
                 name_to_idx: dict):
        self.src = src
        self.dst = dst
        self.mixed_op = mixed_op
        self.name_to_idx = name_to_idx  # op_name → idx in MixedOp.ops

    def has_op(self, op_name: str) -> bool:
        return op_name in self.name_to_idx

    def add_op(self, op_name: str, module: nn.Module):
        """Aggiunge una nuova op al MixedOp."""
        idx = len(self.mixed_op.ops)
        self.mixed_op.ops.append(module)
        self.name_to_idx[op_name] = idx

    def __repr__(self) -> str:
        return f"SuperEdge({self.src!r}→{self.dst!r}, {self.mixed_op})"


# ── Supernet ─────────────────────────────────────────────────────────────────

class Supernet(nn.Module):

    def __init__(self):
        super().__init__()
        self._built = False

        self._topo_order: list = []
        self._nodes: dict = {}
        self._super_edges: list = []
        self._edge_index: dict = {}   # (src, dst) → SuperEdge
        self._in_edges: dict = {}     # node_id → list[SuperEdge]
        self._out_edges: dict = {}    # node_id → list[dst_id]  (per topo sort)
        self._known_paths: dict = {}  # ArchPath → routing dict

        self._all_mixed_ops = nn.ModuleList()
        self.stem: Optional[nn.Module] = None
        self.head: Optional[nn.Module] = None

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_path(net: NetworkDAG) -> ArchPath:
        """Estrae il percorso canonico come frozenset di (src, dst, op_name)."""
        assert len(net.cell_nodes) == 1, "strutture multi-cella non supportate"
        dag = net.cell_nodes[0].cell_spec.dag
        path = set()
        for edge in dag.edges:
            op_name = getattr(edge.op, "op_name", None)
            assert op_name is not None, (
                f"Edge {edge.src}→{edge.dst}: la factory deve avere op_name"
            )
            path.add((edge.src, edge.dst, op_name))
        return frozenset(path)

    @staticmethod
    def _path_to_routing(path: ArchPath) -> dict:
        """Converte un ArchPath in un routing dict: (src, dst) → op_name."""
        return {(src, dst): op_name for src, dst, op_name in path}

    def _assert_built(self):
        assert self._built, "Chiama build() prima di usare la supernet."

    # ── Build / Expand ───────────────────────────────────────────────────────

    def build(self, networks: list, n_classes: int = 10) -> "Supernet":
        """Costruisce la supernet dal primo gruppo di reti."""
        assert not self._built, "La supernet è già stata costruita."
        assert networks, "Serve almeno una rete per costruire la supernet."

        C = networks[0].cell_nodes[0].C_in
        self.stem = Stem(C_out=C)
        self.head = Head(C_in=C, n_classes=n_classes)
        self._built = True
        self._incorporate(networks)
        return self

    def expand(self, networks: list):
        """Espande la supernet con nuove reti non ancora viste."""
        self._assert_built()
        self._incorporate(networks)

    def _incorporate(self, networks: list):
        """Aggiunge i NetworkDAG alla supernet, espandendo archi/op se necessario."""
        for net in networks:
            path = self._extract_path(net)
            if path in self._known_paths:
                continue

            self._known_paths[path] = self._path_to_routing(path)

            cell = net.cell_nodes[0]
            dag = cell.cell_spec.dag
            C_in = cell.C_in
            C_out = cell.C_out

            for node in dag.nodes:
                if node.id not in self._nodes:
                    self._nodes[node.id] = Node(node.id, node.aggregation)
                    self._in_edges[node.id] = []
                    self._out_edges[node.id] = []

            for edge in dag.edges:
                key = (edge.src, edge.dst)
                op_name = edge.op.op_name
                module = edge.op(C_in, C_out)
                module.op_name = op_name

                if key not in self._edge_index:
                    mixed = MixedOp([module])
                    self._all_mixed_ops.append(mixed)
                    se = SuperEdge(edge.src, edge.dst, mixed,
                                   name_to_idx={op_name: 0})
                    self._super_edges.append(se)
                    self._edge_index[key] = se
                    self._in_edges[edge.dst].append(se)
                    self._out_edges[edge.src].append(edge.dst)
                else:
                    se = self._edge_index[key]
                    if not se.has_op(op_name):
                        se.add_op(op_name, module)

            self._recompute_topo()

    # ── Ordine topologico ────────────────────────────────────────────────────

    def _recompute_topo(self):
        in_deg = {nid: 0 for nid in self._nodes}
        for se in self._super_edges:
            in_deg[se.dst] += 1

        queue = [nid for nid, d in in_deg.items() if d == 0]
        topo = []
        while queue:
            nid = queue.pop(0)
            topo.append(nid)
            for dst in self._out_edges[nid]:
                in_deg[dst] -= 1
                if in_deg[dst] == 0:
                    queue.append(dst)

        assert len(topo) == len(self._nodes), "Ciclo rilevato nella supernet."
        self._topo_order = topo

    # ── Forward di una subnet ────────────────────────────────────────────────

    def _forward_subnet(self, x: torch.Tensor, routing: dict) -> torch.Tensor:
        """Forward della subnet identificata dal routing (src,dst) → op_name."""
        x = self.stem(x)
        node_out: dict = {}

        for nid in self._topo_order:
            node = self._nodes[nid]
            active = [
                (se, se.name_to_idx[routing[(se.src, se.dst)]])
                for se in self._in_edges[nid]
                if (se.src, se.dst) in routing
            ]

            if not active:
                node_out[nid] = x
                continue

            partial = [se.mixed_op(node_out[se.src], idx) for se, idx in active]

            if node.aggregation == "sum":
                node_out[nid] = sum(partial[1:], partial[0])
            elif node.aggregation == "mean":
                node_out[nid] = sum(partial[1:], partial[0]) / len(partial)
            elif node.aggregation == "concat":
                node_out[nid] = torch.cat(partial, dim=1)
            else:
                raise ValueError(f"Aggregazione sconosciuta: {node.aggregation}")

        return self.head(node_out[self._topo_order[-1]])

    # ── BN Calibration + Eval ────────────────────────────────────────────────

    def _calibrate_bn(self, routing: dict, loader, device: str,
                      n_batches: int = 50):
        """Resetta e ricalibra le BatchNorm sulla subnet data."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
        self.train()
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                if i >= n_batches:
                    break
                self._forward_subnet(x.to(device), routing)

    def _eval_one(self, routing: dict, loader, device: str) -> float:
        """Valuta l'accuracy top-1 della subnet sul loader dato."""
        self.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = self._forward_subnet(x, routing).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

    # ── Training della supernet ──────────────────────────────────────────────

    def _train_supernet(
        self,
        train_loader,
        epochs: int = 120,
        start_epoch: int = 0,
        M: int = 4,
        criterion=None,
        use_label_smoothing: bool = False,
        scheduler_factory=None,
        optimizer_factory=None,
    ) -> dict:
        """
        Training con random path sampling (stile SPOS).
        Ad ogni batch campiona M path e accumula il gradiente su tutti.
        """
        device = next(self.parameters()).device

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        if optimizer_factory is not None:
            optimizer = optimizer_factory(self.parameters())
        else:
            optimizer = torch.optim.SGD(
                self.parameters(), lr=0.025,
                momentum=0.9, weight_decay=3e-4, nesterov=True,
            )

        if scheduler_factory is not None:
            scheduler = scheduler_factory(optimizer)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-4,
                last_epoch=start_epoch - 1,  # riprende correttamente dal checkpoint
            )

        known_paths = list(self._known_paths.values())
        assert known_paths, "Nessun path noto — chiama build() prima."

        checkpoint_state = {}

        for epoch in range(start_epoch, epochs):
            self.train()

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()

                batch_loss = 0.0
                for routing in random.choices(known_paths, k=M):
                    loss = criterion(self._forward_subnet(inputs, routing), targets) / M
                    loss.backward()
                    batch_loss += loss.item()

                # azzera gradienti delle op "zero" per non inquinare il passo
                for p in self.parameters():
                    if p.grad is not None and p.grad.abs().sum() == 0:
                        p.grad = None

                if not use_label_smoothing:
                    nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)

                optimizer.step()

                if batch_idx % 25 == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    print(f"Epoch {epoch:3d} | Batch {batch_idx:4d} | "
                          f"Loss: {batch_loss:.4f} | LR: {lr:.6f} | "
                          f"Path noti: {len(known_paths)}")

            scheduler.step()

            checkpoint_state = {
                "epoch": epoch,
                "state_dict": self.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "known_paths": list(self._known_paths.keys()),
            }

        return checkpoint_state

    # ── API pubblica ─────────────────────────────────────────────────────────

    def eval_subnets(
        self,
        networks: list,
        train_loader,
        eval_loader,
        device: str = "cuda",
        bn_batches: int = 50,
        n_classes: int = 10,
        epochs: int = 120,
        start_epoch: int = 0,
        M: int = 4,
        criterion=None,
        use_label_smoothing: bool = False,
        scheduler_factory=None,
        optimizer_factory=None,
        calibrate: bool = False,
    ) -> list:
        """
        Costruisce (o espande) la supernet, la allena e restituisce
        l'accuracy di ogni subnet in `networks`.
        """
        if not self._built:
            self.build(networks, n_classes=n_classes)
        else:
            self.expand(networks)

        self.to(device)

        self._train_supernet(
            train_loader,
            epochs=epochs,
            start_epoch=start_epoch,
            M=M,
            criterion=criterion,
            use_label_smoothing=use_label_smoothing,
            scheduler_factory=scheduler_factory,
            optimizer_factory=optimizer_factory,
        )

        accuracies = []
        print("evaluating networks\n")
        for net in networks:
            path = self._extract_path(net)
            routing = self._known_paths[path]
            if calibrate:
                self._calibrate_bn(routing, train_loader, device, bn_batches)
            acc = self._eval_one(routing, eval_loader, device)
            accuracies.append(acc)

        return accuracies

    # ── Utilità ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        if not self._built:
            return "Supernet(non costruita)"
        n_ops = sum(len(se.mixed_op.ops) for se in self._super_edges)
        return (f"Supernet({len(self._known_paths)} subnet note, "
                f"{len(self._super_edges)} archi, {n_ops} op totali)")

    def summary(self):
        print(self)
        for se in self._super_edges:
            print(f"  {se}")
