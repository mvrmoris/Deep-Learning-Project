# supernet.py
import torch
import torch.nn as nn
from network_encoding import (
    OpFactory, Node, Edge, DAG,
    CellSpec, CellNode, CellEdge, NetworkDAG,
)


# ── tipo interno per identificare un percorso ─────────────────────────────────

ArchPath = frozenset  # frozenset[tuple[str, str, str]] = {(src, dst, op_name)}

# ── Stem e Head come moduli fissi ─────────────────────────────────────────────

class Stem(nn.Module):
    """Conv3x3 + BN + ReLU: porta da 3 canali a C canali."""
    def __init__(self, C_out: int = 16):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(3, C_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.op(x)


class Head(nn.Module):
    """Global Average Pooling + Linear classifier."""
    def __init__(self, C_in: int = 16, n_classes: int = 10):
        super().__init__()
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_in, n_classes)

    def forward(self, x):
        x = self.gap(x)
        x = x.flatten(1)
        return self.classifier(x)


def _extract_path(net: NetworkDAG) -> ArchPath:
    """
    Estrae il percorso canonico di un NetworkDAG come frozenset
    di tuple (src, dst, op_name). Questo è l'identificatore univoco
    dell'architettura all'interno della supernet.
    """
    assert len(net.cell_nodes) == 1, "strutture multi-cella non ancora supportate"
    dag = net.cell_nodes[0].cell_spec.dag
    path = set()
    for edge in dag.edges:
        op_name = getattr(edge.op, "op_name", None)
        assert op_name is not None, (
            f"Edge {edge.src}→{edge.dst}: la factory deve avere op_name"
        )
        path.add((edge.src, edge.dst, op_name))
    return frozenset(path)


# ── MixedOp ───────────────────────────────────────────────────────────────────

class MixedOp(nn.Module):
    """
    Contenitore di operazioni su un arco della supernet.
    Ogni op è istanziata una sola volta — weight sharing garantito.
    """
    def __init__(self, ops: list[nn.Module]):
        super().__init__()
        self.ops = nn.ModuleList(ops)

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return self.ops[idx](x)

    def op_names(self) -> list[str]:
        return [getattr(op, "op_name", type(op).__name__) for op in self.ops]

    def __repr__(self):
        return f"MixedOp({self.op_names()})"


# ── SuperEdge ─────────────────────────────────────────────────────────────────

class SuperEdge:
    """
    Arco nella supernet.
    name_to_idx: op_name → indice in MixedOp.ops
    """
    def __init__(self, src: str, dst: str, mixed_op: MixedOp,
                 name_to_idx: dict[str, int]):
        self.src         = src
        self.dst         = dst
        self.mixed_op    = mixed_op
        self.name_to_idx = name_to_idx   # op_name → idx

    def has_op(self, op_name: str) -> bool:
        return op_name in self.name_to_idx

    def add_op(self, op_name: str, module: nn.Module):
        """Espande il MixedOp con una nuova op (usato da _expand)."""
        idx = len(self.mixed_op.ops)
        self.mixed_op.ops.append(module)
        self.name_to_idx[op_name] = idx

    def __repr__(self):
        return f"SuperEdge({self.src!r}→{self.dst!r}, {self.mixed_op})"


# ── Supernet ──────────────────────────────────────────────────────────────────

class Supernet(nn.Module):

    def __init__(self):
        super().__init__()
        self._built         = False
        self._topo_order    : list[str]                          = []
        self._nodes         : dict[str, Node]                   = {}
        self._super_edges   : list[SuperEdge]                   = []
        self._edge_index    : dict[tuple[str,str], SuperEdge]   = {}
        self._in_edges      : dict[str, list[SuperEdge]]        = {}
        self._out_edges_tmp : dict[str, list[str]]              = {}
        self._input_nodes   : list[str]                         = []
        self._output_nodes  : list[str]                         = []
        self._known_paths   : dict[ArchPath, dict[tuple[str,str], str]] = {}
        self._all_mixed_ops = nn.ModuleList()

        # stem e head: istanziati in build quando conosciamo C e n_classes
        self.stem : nn.Module | None = None
        self.head : nn.Module | None = None


    # ── path → routing dict ───────────────────────────────────────────────────

    @staticmethod
    def _path_to_routing(path: ArchPath) -> dict[tuple[str,str], str]:
        """(src,dst) → op_name per un dato percorso."""
        return {(src, dst): op_name for src, dst, op_name in path}


    # ── build ─────────────────────────────────────────────────────────────────

    def build(self, networks: list[NetworkDAG],
              n_classes: int = 10) -> "Supernet":
        assert not self._built
        assert len(networks) > 0

        # ricava C dal primo CellNode
        C = networks[0].cell_nodes[0].C_in

        self.stem = Stem(C_out=C)
        self.head = Head(C_in=C, n_classes=n_classes)

        self._built = True
        self._incorporate(networks)
        return self


    # ── _incorporate: logica comune a build ed _expand ────────────────────────

    def _incorporate(self, networks: list[NetworkDAG]):
        """
        Aggiunge i NetworkDAG alla supernet:
        - se una rete è già nota (stesso path) la salta
        - se ha archi nuovi o op nuove su archi esistenti, espande
        """
        for net in networks:
            path = _extract_path(net)

            if path in self._known_paths:
                continue  # già presente, niente da fare

            routing = self._path_to_routing(path)
            self._known_paths[path] = routing

            # estrai dag interno
            cell      = net.cell_nodes[0]
            dag       = cell.cell_spec.dag
            C_in      = cell.C_in
            C_out     = cell.C_out

            # ── aggiungi nodi mancanti ─────────────────────────────────────
            for node in dag.nodes:
                if node.id not in self._nodes:
                    self._nodes[node.id] = Node(node.id, node.aggregation)
                    self._in_edges[node.id]    = []
                    self._out_edges_tmp[node.id] = []

            # ── aggiungi/espandi archi ─────────────────────────────────────
            for edge in dag.edges:
                key      = (edge.src, edge.dst)
                op_name  = edge.op.op_name
                module   = edge.op(C_in, C_out)
                if hasattr(edge.op, "op_name"):
                    module.op_name = edge.op.op_name

                if key not in self._edge_index:
                    # arco completamente nuovo
                    mixed = MixedOp([module])
                    self._all_mixed_ops.append(mixed)
                    se = SuperEdge(edge.src, edge.dst, mixed,
                                   name_to_idx={op_name: 0})
                    self._super_edges.append(se)
                    self._edge_index[key] = se
                    self._in_edges[edge.dst].append(se)
                    self._out_edges_tmp[edge.src].append(edge.dst)

                else:
                    # arco esistente: aggiungi op se non c'è già
                    se = self._edge_index[key]
                    if not se.has_op(op_name):
                        se.add_op(op_name, module)

            # aggiorna ordine topologico dopo ogni rete
            self._recompute_topo()


    # ── _expand ───────────────────────────────────────────────────────────────

    def expand(self, networks: list[NetworkDAG]):
        """
        Espande la supernet con nuove reti non ancora viste.
        Chiamato automaticamente da eval_subnets se necessario.
        """
        assert self._built, "chiama build prima"
        self._incorporate(networks)


    # ── ordine topologico ─────────────────────────────────────────────────────

    def _recompute_topo(self):
        in_deg = {nid: 0 for nid in self._nodes}
        for se in self._super_edges:
            in_deg[se.dst] += 1
        q    = [nid for nid, d in in_deg.items() if d == 0]
        topo = []
        while q:
            nid = q.pop(0); topo.append(nid)
            for dst in self._out_edges_tmp[nid]:
                in_deg[dst] -= 1
                if in_deg[dst] == 0: q.append(dst)
        assert len(topo) == len(self._nodes), "ciclo nella supernet"
        self._topo_order    = topo
        self._input_nodes   = [nid for nid in self._nodes
                                if not self._in_edges[nid]]
        self._output_nodes  = [nid for nid in topo
                                if not self._out_edges_tmp[nid]]


    # ── forward di una subnet dato il routing ────────────────────────────────

    def _forward_subnet(self, x: torch.Tensor,
                        routing: dict[tuple[str,str], str]) -> torch.Tensor:
        """
        Esegue il forward della subnet identificata da routing.
        routing: (src,dst) → op_name
        """
        # ── stem: 3 canali → C canali ─────────────────────────────────────────
        x = self.stem(x)

        node_tensors: dict[str, torch.Tensor] = {}

        for nid in self._topo_order:
            node     = self._nodes[nid]
            incoming = self._in_edges[nid]

            # archi attivi per questo routing
            active = []
            for se in incoming:
                key = (se.src, se.dst)
                if key in routing:
                    op_name = routing[key]
                    idx     = se.name_to_idx[op_name]
                    active.append((se, idx))

            if not active:
                node_tensors[nid] = x
            else:
                partial = [se.mixed_op(node_tensors[se.src], idx)
                           for se, idx in active]

                if node.aggregation == "sum":
                    out = partial[0]
                    for t in partial[1:]: out = out + t
                elif node.aggregation == "mean":
                    out = partial[0]
                    for t in partial[1:]: out = out + t
                    out = out / len(partial)
                elif node.aggregation == "concat":
                    out = torch.cat(partial, dim=1)
                else:
                    raise ValueError(f"aggregazione sconosciuta: {node.aggregation}")

                node_tensors[nid] = out

        # ── head: GAP + Linear → logits ───────────────────────────────────────
        return self.head(node_tensors[self._topo_order[-1]])


    # ── BN calibration ────────────────────────────────────────────────────────

    def _calibrate_bn(self, routing: dict[tuple[str,str], str],
                      loader, device: str, n_batches: int = 50):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
        self.train()
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                if i >= n_batches: break
                self._forward_subnet(x.to(device), routing)


    def _eval_one(self, routing: dict[tuple[str,str], str],
                  loader, device: str) -> float:
        self.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out  = self._forward_subnet(x, routing)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total   += y.size(0)
        return correct / total if total > 0 else 0.0


    # ── _train_supernet (placeholder) ─────────────────────────────────────────

    # ── _train_supernet ────────────────────────────────────────────────────────────

    def _train_supernet(
        self,
        train_loader,
        epochs            : int   = 120,
        start_epoch       : int   = 0,
        M                 : int   = 4,
        criterion                 = None,
        use_label_smoothing: bool = False,
        # scheduler factory: callable che riceve optimizer e ritorna scheduler
        # es. lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=120)
        scheduler_factory         = None,
        # optimizer factory: callable che riceve params e ritorna optimizer
        # es. lambda p: torch.optim.SGD(p, lr=0.025, momentum=0.9, weight_decay=3e-4)
        optimizer_factory         = None,
    ) -> dict:
        """
        Training della supernet con random path sampling (stile SPOS).

        Ad ogni batch:
        - campiona M path tra quelli noti in _known_paths
        - per ogni path esegue il forward, calcola loss, accumula gradiente
        - aggiorna i pesi condivisi

        Parametri visibili allo user:
        epochs, start_epoch     — budget di training
        M                       — path campionati per batch (Monte Carlo)
        criterion               — loss function (default CrossEntropy)
        use_label_smoothing     — se True disabilita grad clipping
        scheduler_factory       — come costruire lo scheduler (opzionale)
        optimizer_factory       — come costruire l'optimizer (opzionale,
                                    default SGD cosine come in NAS-Bench-201)

        Ritorna checkpoint_state dell'ultima epoca.
        """
        import random as _random

        device = next(self.parameters()).device

        # ── criterion ─────────────────────────────────────────────────────────────
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # ── optimizer: costruito sui parametri INTERNI della supernet ─────────────
        if optimizer_factory is not None:
            optimizer = optimizer_factory(self.parameters())
        else:
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=0.025,
                momentum=0.9,
                weight_decay=3e-4,
                nesterov=True,
            )

        # ── scheduler ─────────────────────────────────────────────────────────────
        if scheduler_factory is not None:
            scheduler = scheduler_factory(optimizer)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=1e-4,
            )
            # avanza lo scheduler fino a start_epoch se riprendiamo da checkpoint
            for _ in range(start_epoch):
                scheduler.step()

        # ── lista di path campionabili ─────────────────────────────────────────────
        known_paths    = list(self._known_paths.values())  # list[dict[(src,dst)->op_name]]
        n_known        = len(known_paths)
        assert n_known > 0, "nessun path noto — chiama build prima di _train_supernet"

        checkpoint_state = {}

        # ── loop di training ───────────────────────────────────────────────────────
        for epoch in range(start_epoch, epochs):
            self.train()

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs  = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()

                batch_loss = 0.0

                # campiona M path (con rimpiazzo se M > n_known)
                sampled_routings = _random.choices(known_paths, k=M)

                for routing in sampled_routings:
                    outputs = self._forward_subnet(inputs, routing)
                    loss    = criterion(outputs, targets) / M
                    loss.backward()
                    batch_loss += loss.item()

                # rimuovi gradienti esattamente zero (op "none" / zero-op)
                for p in self.parameters():
                    if p.grad is not None and p.grad.abs().sum() == 0:
                        p.grad = None

                # grad clipping (disabilitato con label smoothing, come nell'originale)
                if not use_label_smoothing:
                    nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)

                optimizer.step()

                if batch_idx % 25 == 0:
                    current_lr = (
                        scheduler.get_last_lr()[0]
                        if scheduler is not None
                        else optimizer.param_groups[0]["lr"]
                    )
                    print(
                        f"Epoch {epoch:3d} | "
                        f"Batch {batch_idx:4d} | "
                        f"Loss (M={M}): {batch_loss:.4f} | "
                        f"LR: {current_lr:.6f} | "
                        f"Path noti: {n_known}"
                    )

            if scheduler is not None:
                scheduler.step()

            checkpoint_state = {
                "epoch"           : epoch,
                "state_dict"      : self.state_dict(),
                "optimizer_state" : optimizer.state_dict(),
                "scheduler_state" : scheduler.state_dict() if scheduler is not None else None,
                "known_paths"     : list(self._known_paths.keys()),
            }

        return checkpoint_state
    # ── eval: unica API pubblica ───────────────────────────────────────────────
    def eval_subnets(
        self,
        networks            : list[NetworkDAG],
        train_loader,
        eval_loader,
        device              : str  = "cuda",
        bn_batches          : int  = 50,
        n_classes           : int  = 10,      # ← nuovo
        epochs              : int  = 120,
        start_epoch         : int  = 0,
        M                   : int  = 4,
        criterion                   = None,
        use_label_smoothing : bool  = False,
        scheduler_factory           = None,
        optimizer_factory           = None,
    ) -> list[float]:

        if not self._built:
            self.build(networks, n_classes=n_classes)
        else:
            self.expand(networks)

        self.to(device)

        self._train_supernet(
            train_loader,
            epochs              = epochs,
            start_epoch         = start_epoch,
            M                   = M,
            criterion           = criterion,
            use_label_smoothing = use_label_smoothing,
            scheduler_factory   = scheduler_factory,
            optimizer_factory   = optimizer_factory,
        )

        accuracies = []
        for net in networks:
            path    = _extract_path(net)
            routing = self._known_paths[path]
            self._calibrate_bn(routing, train_loader, device, bn_batches)
            acc = self._eval_one(routing, eval_loader, device)
            accuracies.append(acc)

        return accuracies


    # ── utilità ───────────────────────────────────────────────────────────────

    def __repr__(self):
        if not self._built:
            return "Supernet(non costruita)"
        n_ops = sum(len(se.mixed_op.ops) for se in self._super_edges)
        return (f"Supernet("
                f"{len(self._known_paths)} subnet note, "
                f"{len(self._super_edges)} archi, "
                f"{n_ops} op totali)")

    def summary(self):
        print(self)
        for se in self._super_edges:
            print(f"  {se}")