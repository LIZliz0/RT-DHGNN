import os
import random
import pickle
import warnings
import argparse
import csv

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report
)

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HGTConv

warnings.filterwarnings("ignore")

# Configuration
GRAPH_PATH = "result/hetero_graph.pt"
META_PATH = "result/meta.pkl"

RESULT_DIR = "result12wc/ablation"
CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")
CSV_RESULT = os.path.join(RESULT_DIR, "ablation_result.csv")

SEED = 42

BATCH_SIZE = 4096

NUM_WORKERS = 0

HIDDEN_DIM = 128
HEADS = 4
NUM_LAYERS = 2

LR = 1e-3
WEIGHT_DECAY = 1e-4

EPOCHS = 50
PATIENCE = 10

LABEL_SMOOTHING = 0.05

GRAD_CLIP = 2.0

EXPERIMENT = "full"

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("RT-HG Ablation Experiments")
print("=" * 70)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# Seed
def seed_all(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_all(SEED)

# Experiment configuration
EXPERIMENT_CONFIG = {

    "full": {
        "name": "RT-HG",
        "use_role": True,
        "use_exec": True,
        "use_rt": True,
        "use_hop": True,
        "use_time": True,
        "backbone": "HGT"
    },

    "wo_role": {
        "name": "RT-HG w/o Role",
        "use_role": False,
        "use_exec": False,
        "use_rt": False,
        "use_hop": False,
        "use_time": True,
        "backbone": "HGT"
    },

    "wo_rt": {
        "name": "RT-HG w/o RT",
        "use_role": True,
        "use_exec": True,
        "use_rt": False,
        "use_hop": True,
        "use_time": True,
        "backbone": "HGT"
    },

    "wo_coupling": {
        "name": "RT-HG w/o Coupling",
        "use_role": True,
        "use_exec": False,
        "use_rt": True,
        "use_hop": True,
        "use_time": True,
        "backbone": "HGT"
    },

    "wo_hop": {
        "name": "RT-HG w/o Hop",
        "use_role": True,
        "use_exec": True,
        "use_rt": True,
        "use_hop": False,
        "use_time": True,
        "backbone": "HGT"
    },

    "wo_time": {
        "name": "RT-HG w/o Time",
        "use_role": True,
        "use_exec": True,
        "use_rt": True,
        "use_hop": True,
        "use_time": False,
        "backbone": "HGT"
    },

    "lite": {
        "name": "RT-HG-Lite",
        "use_role": True,
        "use_exec": True,
        "use_rt": True,
        "use_hop": True,
        "use_time": True,
        "backbone": "RGCN"
    }
}

# Load graph
print("\n" + "=" * 70)
print("Loading Graph")
print("=" * 70)

if not os.path.exists(GRAPH_PATH):
    raise FileNotFoundError(
        f"Graph not found:\n{GRAPH_PATH}"
    )

if not os.path.exists(META_PATH):
    raise FileNotFoundError(
        f"Meta not found:\n{META_PATH}"
    )

data = torch.load(
    GRAPH_PATH,
    weights_only=False
)

with open(META_PATH, "rb") as f:
    meta = pickle.load(f)


NUM_CLASSES = int(meta["num_classes"])
NUM_ACTIVITIES = NUM_CLASSES

NUM_HOPS = int(
    data["event"].hop_id.max().item()
) + 1


print("Num event nodes :", data["event"].num_nodes)
print("Num role nodes  :", data["role"].num_nodes)
print("Num activities  :", NUM_ACTIVITIES)
print("Num hop IDs     :", NUM_HOPS)

# Check graph

print("\nEdge types:")

for edge_type in data.edge_types:

    n_edges = data[edge_type].edge_index.shape[1]

    print(
        f"{edge_type}: {n_edges}"
    )


required_edges = [
    ("event", "historical_follow", "event"),
    ("role", "executes", "event"),
    ("event", "rev_executes", "role"),
    ("role", "role_transition", "role"),
    ("role", "role_transition_rev", "role")
]

for edge_type in required_edges:

    if edge_type not in data.edge_types:

        raise RuntimeError(
            f"Required edge type missing: {edge_type}"
        )


print("\n✔ Graph edge types verified")


# Prediction masks
train_idx = torch.where(
    data["event"].train_mask
)[0]

val_idx = torch.where(
    data["event"].val_mask
)[0]

test_idx = torch.where(
    data["event"].test_mask
)[0]


print("\n" + "=" * 70)
print("Prediction Samples")
print("=" * 70)

print("Train:", len(train_idx))
print("Val  :", len(val_idx))
print("Test :", len(test_idx))

# Validate masks
train_mask = data["event"].train_mask
val_mask = data["event"].val_mask
test_mask = data["event"].test_mask

assert not torch.any(
    train_mask & val_mask
)

assert not torch.any(
    train_mask & test_mask
)

assert not torch.any(
    val_mask & test_mask
)

print("\n✔ Train / Val / Test masks are mutually exclusive")


valid_label_mask = data["event"].y >= 0

print(
    "Valid prediction labels:",
    int(valid_label_mask.sum())
)


# Build filtered graph
def build_experiment_graph(config):

    cfg = EXPERIMENT_CONFIG[config]

    g = HeteroData()

    # Event node

    g["event"].num_nodes = data["event"].num_nodes

    # activity_id
    g["event"].x_int = data["event"].x_int.clone()

    # delta time
    if cfg["use_time"]:

        g["event"].x_float = (
            data["event"].x_float.clone()
        )

    else:

        g["event"].x_float = torch.zeros(
            (data["event"].num_nodes, 1),
            dtype=torch.float
        )

    # hop
    if cfg["use_hop"]:

        g["event"].hop_id = (
            data["event"].hop_id.clone()
        )

    else:

        g["event"].hop_id = torch.zeros(
            data["event"].num_nodes,
            dtype=torch.long
        )

    # label
    g["event"].y = data["event"].y.clone()

    # masks
    g["event"].train_mask = (
        data["event"].train_mask.clone()
    )

    g["event"].val_mask = (
        data["event"].val_mask.clone()
    )

    g["event"].test_mask = (
        data["event"].test_mask.clone()
    )

    # Role node

    if cfg["use_role"]:

        g["role"].num_nodes = (
            data["role"].num_nodes
        )

        g["role"].x = (
            data["role"].x.clone()
        )

    # Historical control flow

    edge_type = (
        "event",
        "historical_follow",
        "event"
    )

    g[edge_type].edge_index = (
        data[edge_type].edge_index.clone()
    )

    # Role execution coupling

    if cfg["use_role"] and cfg["use_exec"]:

        edge_type = (
            "role",
            "executes",
            "event"
        )

        g[edge_type].edge_index = (
            data[edge_type].edge_index.clone()
        )

        edge_type = (
            "event",
            "rev_executes",
            "role"
        )

        g[edge_type].edge_index = (
            data[edge_type].edge_index.clone()
        )

    # Role transition

    if cfg["use_role"] and cfg["use_rt"]:

        edge_type = (
            "role",
            "role_transition",
            "role"
        )

        g[edge_type].edge_index = (
            data[edge_type].edge_index.clone()
        )

        # 保留 role_transition_rev
        edge_type = (
            "role",
            "role_transition_rev",
            "role"
        )

        g[edge_type].edge_index = (
            data[edge_type].edge_index.clone()
        )

        # edge weights
        if hasattr(
            data[("role",
                  "role_transition",
                  "role")],
            "edge_weight"
        ):

            g[
                ("role",
                 "role_transition",
                 "role")
            ].edge_weight = (
                data[
                    ("role",
                     "role_transition",
                     "role")
                ].edge_weight.clone()
            )

        if hasattr(
            data[("role",
                  "role_transition_rev",
                  "role")],
            "edge_weight"
        ):

            g[
                ("role",
                 "role_transition_rev",
                 "role")
            ].edge_weight = (
                data[
                    ("role",
                     "role_transition_rev",
                     "role")
                ].edge_weight.clone()
            )

    return g


# Print experiment configuration
def print_experiment_config(config):

    cfg = EXPERIMENT_CONFIG[config]

    print("\n" + "=" * 70)
    print("Experiment:", cfg["name"])
    print("=" * 70)

    print("Role nodes       :", cfg["use_role"])
    print("Execution edges  :", cfg["use_exec"])
    print("RT edges         :", cfg["use_rt"])
    print("Hop feature      :", cfg["use_hop"])
    print("Time feature     :", cfg["use_time"])
    print("Backbone         :", cfg["backbone"])

# NeighborLoader
def build_loader(
    graph,
    idx,
    shuffle=False
):

    edge_types = graph.edge_types

    num_neighbors = {
        edge_type: [20, 10]
        for edge_type in edge_types
    }

    loader = NeighborLoader(

        graph,

        input_nodes=(
            "event",
            idx
        ),

        num_neighbors=num_neighbors,

        batch_size=BATCH_SIZE,

        shuffle=shuffle,

        num_workers=NUM_WORKERS,

        pin_memory=(
            device.type == "cuda"
        )
    )

    return loader


# HGT Encoder

class HGTModel(nn.Module):

    def __init__(
        self,
        graph,
        use_hop=True,
        use_time=True
    ):

        super().__init__()

        self.use_hop = use_hop
        self.use_time = use_time

        # Activity embedding

        self.act_emb = nn.Embedding(
            NUM_ACTIVITIES,
            64
        )

        # Hop embedding

        if use_hop:

            self.hop_emb = nn.Embedding(
                NUM_HOPS,
                32
            )

        # Time
        if use_time:

            self.delta = nn.Sequential(

                nn.Linear(
                    1,
                    32
                ),

                nn.ReLU()
            )

        # Calculate input dimension
        input_dim = 64

        if use_hop:
            input_dim += 32

        if use_time:
            input_dim += 32

        # Event projection
        self.event_fusion = nn.Sequential(

            nn.Linear(
                input_dim,
                HIDDEN_DIM
            ),

            nn.ReLU()
        )

        # Role projection
        if "role" in graph.node_types:

            self.role = nn.Sequential(

                nn.Linear(
                    2,
                    HIDDEN_DIM
                ),

                nn.ReLU()
            )

        # HGT
        self.convs = nn.ModuleList()

        self.norms = nn.ModuleList()

        for _ in range(NUM_LAYERS):

            self.convs.append(

                HGTConv(

                    in_channels=HIDDEN_DIM,

                    out_channels=HIDDEN_DIM,

                    metadata=graph.metadata(),

                    heads=HEADS
                )
            )

            self.norms.append(

                nn.ModuleDict({

                    node_type:
                    nn.LayerNorm(HIDDEN_DIM)

                    for node_type
                    in graph.node_types

                })
            )

        # Classifier
        self.cls = nn.Sequential(

            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(
                HIDDEN_DIM,
                NUM_CLASSES
            )
        )

    # Forward

    def forward(self, batch):

        # Input feature domain

        with torch.amp.autocast(
            "cuda",
            enabled=(
                device.type == "cuda"
            )
        ):

            act = self.act_emb(
                batch["event"].x_int.squeeze(-1)
            )

            features = [act]

            if self.use_hop:

                hop = self.hop_emb(
                    batch["event"].hop_id
                )

                features.append(hop)

            if self.use_time:

                delta = self.delta(
                    batch["event"].x_float
                )

                features.append(delta)

            x_event = torch.cat(
                features,
                dim=-1
            )

            x_event = self.event_fusion(
                x_event
            )

            x_dict = {
                "event": x_event.float()
            }

            if "role" in batch.node_types:

                x_role = self.role(
                    batch["role"].x
                )

                x_dict["role"] = (
                    x_role.float()
                )

        # HGT
        edge_index_dict = batch.edge_index_dict

        for conv, norm_dict in zip(
            self.convs,
            self.norms
        ):

            residual = {
                key: value
                for key, value
                in x_dict.items()
            }

            with torch.amp.autocast(
                "cuda",
                enabled=False
            ):

                x_dict = conv(
                    x_dict,
                    edge_index_dict
                )

            for node_type in x_dict:

                if node_type in residual:

                    x_dict[node_type] = (
                        norm_dict[node_type](
                            x_dict[node_type]
                            +
                            residual[node_type]
                        )
                    )

                    x_dict[node_type] = F.relu(
                        x_dict[node_type]
                    )

        return self.cls(
            x_dict["event"]
        )

# RGCN-Lite
class RelationLayer(nn.Module):


    def __init__(
        self,
        hidden_dim,
        edge_types
    ):

        super().__init__()

        self.edge_types_list = list(
            edge_types
        )

        self.linears = nn.ModuleDict()

        for i, edge_type in enumerate(
            self.edge_types_list
        ):

            self.linears[str(i)] = nn.Linear(
                hidden_dim,
                hidden_dim,
                bias=False
            )

        self.self_linears = nn.ModuleDict({

            node_type:
            nn.Linear(
                hidden_dim,
                hidden_dim,
                bias=True
            )

            for node_type in
            ["event", "role"]

        })

    def forward(
        self,
        x_dict,
        edge_index_dict
    ):

        out = {}

        for node_type in x_dict:

            out[node_type] = (
                self.self_linears[node_type](
                    x_dict[node_type]
                )
            )

        for rel_id, edge_type in enumerate(
            self.edge_types_list
        ):

            if edge_type not in edge_index_dict:
                continue

            src_type, _, dst_type = edge_type

            edge_index = edge_index_dict[
                edge_type
            ]

            src = edge_index[0]
            dst = edge_index[1]

            x_src = x_dict[src_type]

            messages = self.linears[
                str(rel_id)
            ](x_src[src])

            # Mean aggregation
            aggregated = torch.zeros(

                (
                    x_dict[dst_type].size(0),
                    messages.size(1)
                ),

                device=messages.device,

                dtype=messages.dtype
            )

            aggregated.index_add_(
                0,
                dst,
                messages
            )

            degree = torch.zeros(
                x_dict[dst_type].size(0),
                device=messages.device,
                dtype=messages.dtype
            )

            degree.index_add_(
                0,
                dst,
                torch.ones_like(
                    dst,
                    dtype=messages.dtype
                )
            )

            degree = degree.clamp(
                min=1.0
            ).unsqueeze(-1)

            aggregated = (
                aggregated / degree
            )

            out[dst_type] = (
                out[dst_type]
                +
                aggregated
            )

        return out


class RGCNLiteModel(nn.Module):

    def __init__(
        self,
        graph,
        use_hop=True,
        use_time=True
    ):

        super().__init__()

        self.use_hop = use_hop
        self.use_time = use_time

        # Input features
        self.act_emb = nn.Embedding(
            NUM_ACTIVITIES,
            64
        )

        if use_hop:

            self.hop_emb = nn.Embedding(
                NUM_HOPS,
                32
            )

        if use_time:

            self.delta = nn.Sequential(

                nn.Linear(
                    1,
                    32
                ),

                nn.ReLU()
            )

        input_dim = 64

        if use_hop:
            input_dim += 32

        if use_time:
            input_dim += 32

        self.event_fusion = nn.Sequential(

            nn.Linear(
                input_dim,
                HIDDEN_DIM
            ),

            nn.ReLU()
        )

        # Role
        if "role" in graph.node_types:

            self.role = nn.Sequential(

                nn.Linear(
                    2,
                    HIDDEN_DIM
                ),

                nn.ReLU()
            )

        # RGCN relation layers
        self.layers = nn.ModuleList()

        self.norms = nn.ModuleList()

        for _ in range(NUM_LAYERS):

            self.layers.append(

                RelationLayer(
                    HIDDEN_DIM,
                    graph.edge_types
                )
            )

            self.norms.append(

                nn.ModuleDict({

                    node_type:
                    nn.LayerNorm(
                        HIDDEN_DIM
                    )

                    for node_type
                    in graph.node_types

                })
            )

        # Classifier

        self.cls = nn.Sequential(

            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(
                HIDDEN_DIM,
                NUM_CLASSES
            )
        )

    def forward(self, batch):

        with torch.amp.autocast(
            "cuda",
            enabled=(
                device.type == "cuda"
            )
        ):

            act = self.act_emb(
                batch["event"].x_int.squeeze(-1)
            )

            features = [act]

            if self.use_hop:

                features.append(
                    self.hop_emb(
                        batch["event"].hop_id
                    )
                )

            if self.use_time:

                features.append(
                    self.delta(
                        batch["event"].x_float
                    )
                )

            x_event = torch.cat(
                features,
                dim=-1
            )

            x_event = self.event_fusion(
                x_event
            )

            x_dict = {
                "event": x_event.float()
            }

            if "role" in batch.node_types:

                x_dict["role"] = (
                    self.role(
                        batch["role"].x
                    ).float()
                )

        for layer, norm_dict in zip(
            self.layers,
            self.norms
        ):

            residual = {
                k: v
                for k, v in x_dict.items()
            }

            with torch.amp.autocast(
                "cuda",
                enabled=False
            ):

                x_dict = layer(
                    x_dict,
                    batch.edge_index_dict
                )

            for node_type in x_dict:

                x_dict[node_type] = (
                    norm_dict[node_type](
                        x_dict[node_type]
                        +
                        residual[node_type]
                    )
                )

                x_dict[node_type] = F.relu(
                    x_dict[node_type]
                )

        return self.cls(
            x_dict["event"]
        )

# Model Factory

def create_model(
    graph,
    config
):

    cfg = EXPERIMENT_CONFIG[config]

    if cfg["backbone"] == "HGT":

        return HGTModel(

            graph,

            use_hop=cfg["use_hop"],

            use_time=cfg["use_time"]

        )

    elif cfg["backbone"] == "RGCN":

        return RGCNLiteModel(

            graph,

            use_hop=cfg["use_hop"],

            use_time=cfg["use_time"]

        )

    else:

        raise ValueError(
            f"Unknown backbone: "
            f"{cfg['backbone']}"
        )

# AMP utilities
if device.type == "cuda":

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None

# Train
def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler
):

    model.train()

    total_loss = 0.0

    total_samples = 0

    for batch in loader:

        batch = batch.to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        if device.type == "cuda":

            with torch.amp.autocast(
                "cuda"
            ):

                logits = model(
                    batch
                )

                n = (
                    batch["event"]
                    .batch_size
                )

                logits_target = (
                    logits[:n]
                )

                y = (
                    batch["event"]
                    .y[:n]
                )

                mask = y >= 0

                if mask.sum() == 0:
                    continue

                loss = F.cross_entropy(

                    logits_target[mask],

                    y[mask],

                    label_smoothing=(
                        LABEL_SMOOTHING
                    )
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            logits = model(
                batch
            )

            n = (
                batch["event"]
                .batch_size
            )

            logits_target = (
                logits[:n]
            )

            y = (
                batch["event"]
                .y[:n]
            )

            mask = y >= 0

            if mask.sum() == 0:
                continue

            loss = F.cross_entropy(

                logits_target[mask],

                y[mask],

                label_smoothing=(
                    LABEL_SMOOTHING
                )
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            optimizer.step()

        total_loss += (
            loss.item()
            *
            int(mask.sum())
        )

        total_samples += (
            int(mask.sum())
        )

    if total_samples == 0:
        return 0.0

    return (
        total_loss
        /
        total_samples
    )

# Evaluation
@torch.no_grad()
def evaluate(
    model,
    loader,
    num_classes
):

    model.eval()

    y_true = []
    y_pred = []

    for batch in loader:

        batch = batch.to(
            device
        )

        # Evaluation FP32
        with torch.amp.autocast(
            "cuda",
            enabled=False
        ):

            logits = model(
                batch
            )

        n = (
            batch["event"]
            .batch_size
        )

        logits = logits[:n]

        y = (
            batch["event"]
            .y[:n]
        )

        mask = y >= 0

        if mask.sum() == 0:
            continue

        pred = logits.argmax(
            dim=1
        )

        y_true.extend(
            y[mask]
            .cpu()
            .tolist()
        )

        y_pred.extend(
            pred[mask]
            .cpu()
            .tolist()
        )

    acc = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(

            y_true,

            y_pred,

            labels=list(
                range(num_classes)
            ),

            zero_division=0
        )
    )

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "y_true": y_true,
        "y_pred": y_pred
    }

# Run one experiment
def run_experiment(
    config
):

    cfg = EXPERIMENT_CONFIG[config]

    print_experiment_config(
        config
    )

    # Build graph
    graph = build_experiment_graph(
        config
    )

    print("\nGraph for this experiment:")

    print(
        "Node types:",
        graph.node_types
    )

    print(
        "Edge types:"
    )

    for edge_type in graph.edge_types:

        print(
            " ",
            edge_type,
            ":",
            graph[
                edge_type
            ].edge_index.shape[1]
        )

    # Check w/o role
    if config == "wo_role":

        assert "role" not in graph.node_types

        for edge_type in graph.edge_types:

            assert "role" not in edge_type

        print(
            "\n✔ w/o Role graph verified"
        )

    # Check w/o RT
    if config == "wo_rt":

        assert (
            "role",
            "role_transition",
            "role"
        ) not in graph.edge_types

        assert (
            "role",
            "role_transition_rev",
            "role"
        ) not in graph.edge_types

        print(
            "\n✔ w/o RT graph verified"
        )

    # Check w/o Coupling
    if config == "wo_coupling":

        assert (
            "role",
            "executes",
            "event"
        ) not in graph.edge_types

        assert (
            "event",
            "rev_executes",
            "role"
        ) not in graph.edge_types

        assert (
            "role",
            "role_transition",
            "role"
        ) in graph.edge_types

        assert (
            "role",
            "role_transition_rev",
            "role"
        ) in graph.edge_types

        print(
            "\n✔ w/o Coupling graph verified"
        )

    # Build loaders
    print(
        "\nBuilding NeighborLoaders..."
    )

    train_loader = build_loader(
        graph,
        train_idx,
        shuffle=True
    )

    val_loader = build_loader(
        graph,
        val_idx,
        shuffle=False
    )

    test_loader = build_loader(
        graph,
        test_idx,
        shuffle=False
    )

    print(
        "✔ NeighborLoaders created"
    )

    # Model
    model = create_model(
        graph,
        config
    ).to(device)

    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\nModel parameters:",
        num_params
    )

    # Optimizer
    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LR,

        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS
        )
    )

    # Checkpoint
    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"{config}_best.pt"
    )

    # Training
    best_f1 = -1.0
    best_acc = 0.0
    best_epoch = -1
    patience_count = 0

    history = []

    for epoch in range(
        EPOCHS
    ):

        loss = train_one_epoch(

            model,

            train_loader,

            optimizer,

            scaler
        )

        val_result = evaluate(

            model,

            val_loader,

            NUM_CLASSES
        )

        scheduler.step()

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print(

            f"Epoch {epoch:02d} | "
            f"Loss {loss:.6f} | "
            f"Val Acc "
            f"{val_result['acc']:.6f} | "
            f"Val F1 "
            f"{val_result['macro_f1']:.6f} | "
            f"LR {current_lr:.7f}"

        )

        history.append({

            "epoch": epoch,

            "loss": loss,

            "val_acc":
                val_result["acc"],

            "val_macro_f1":
                val_result["macro_f1"],

            "val_weighted_f1":
                val_result["weighted_f1"],

            "lr": current_lr

        })

        # Best checkpoint
        if (
            val_result["macro_f1"]
            >
            best_f1
        ):

            best_f1 = (
                val_result[
                    "macro_f1"
                ]
            )

            best_acc = (
                val_result[
                    "acc"
                ]
            )

            best_epoch = epoch

            patience_count = 0

            torch.save(

                {
                    "model_state_dict":
                        model.state_dict(),

                    "config":
                        config,

                    "epoch":
                        epoch,

                    "val_acc":
                        best_acc,

                    "val_macro_f1":
                        best_f1,

                    "num_classes":
                        NUM_CLASSES,

                    "num_hops":
                        NUM_HOPS

                },

                checkpoint_path

            )

            print(
                "✔ Saved Best Model"
            )

        else:

            patience_count += 1

        if (
            patience_count
            >= PATIENCE
        ):

            print(
                "Early Stop"
            )

            break

    # Load best
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # Final validation
    final_val = evaluate(

        model,

        val_loader,

        NUM_CLASSES
    )

    # Final test
    final_test = evaluate(

        model,

        test_loader,

        NUM_CLASSES
    )

    # Print result
    print("\n" + "=" * 70)
    print(
        "FINAL RESULT:",
        cfg["name"]
    )
    print("=" * 70)

    print(
        "Best Epoch:",
        checkpoint["epoch"]
    )

    print(
        "Best Val Acc:",
        final_val["acc"]
    )

    print(
        "Best Val Macro-F1:",
        final_val["macro_f1"]
    )

    print(
        "Test Accuracy:",
        final_test["acc"]
    )

    print(
        "Test Macro-F1:",
        final_test["macro_f1"]
    )

    print(
        "Test Weighted-F1:",
        final_test["weighted_f1"]
    )

    # Classification report
    print(
        "\nClassification Report:"
    )

    print(
        classification_report(

            final_test["y_true"],

            final_test["y_pred"],

            labels=list(
                range(NUM_CLASSES)
            ),

            target_names=[
                str(x)
                for x in meta[
                    "activity_vocab"
                ]
            ],

            zero_division=0
        )
    )

    # Return
    result = {

        "experiment":
            config,

        "name":
            cfg["name"],

        "backbone":
            cfg["backbone"],

        "best_epoch":
            checkpoint["epoch"],

        "val_acc":
            final_val["acc"],

        "val_macro_f1":
            final_val[
                "macro_f1"
            ],

        "val_weighted_f1":
            final_val[
                "weighted_f1"
            ],

        "test_acc":
            final_test[
                "acc"
            ],

        "test_macro_f1":
            final_test[
                "macro_f1"
            ],

        "test_weighted_f1":
            final_test[
                "weighted_f1"
            ],

        "parameters":
            num_params,

        "checkpoint":
            checkpoint_path

    }

    # Save history
    history_path = os.path.join(
        RESULT_DIR,
        f"{config}_history.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False
    )

    # Save class-level result
    class_result = pd.DataFrame({

        "activity":
            meta["activity_vocab"],

        "precision":
            final_test["precision"],

        "recall":
            final_test["recall"],

        "f1":
            final_test["f1"],

        "support":
            final_test["support"]

    })

    class_path = os.path.join(
        RESULT_DIR,
        f"{config}_class_metrics.csv"
    )

    class_result.to_csv(
        class_path,
        index=False
    )

    return result

# Save summary
def save_summary(result):

    if len(result) == 0:
        return

    df_result = pd.DataFrame(
        result
    )

    df_result.to_csv(
        CSV_RESULT,
        index=False
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ABLATION SUMMARY"
    )

    print(
        "=" * 70
    )

    display_cols = [

        "name",

        "backbone",

        "test_acc",

        "test_macro_f1",

        "test_weighted_f1",

        "parameters"

    ]

    print(
        df_result[
            display_cols
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        CSV_RESULT
    )

# Main
def main():

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )
    # Parse optional command line
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        default=EXPERIMENT,
        choices=[
            "full",
            "wo_role",
            "wo_rt",
            "wo_coupling",
            "wo_hop",
            "wo_time",
            "lite",
            "all"
        ]
    )

    args = parser.parse_args()

    experiment = args.experiment

    # Determine experiments
    if experiment == "all":

        experiments = [

            "full",

            "wo_role",

            "wo_rt",

            "wo_coupling",

            "wo_hop",

            "wo_time",

            "lite"

        ]

    else:

        experiments = [
            experiment
        ]

    print(
        "\nExperiments to run:"
    )

    for exp in experiments:

        print(
            "  -",
            EXPERIMENT_CONFIG[
                exp
            ]["name"]
        )

    # Run
    result = []

    for i, exp in enumerate(
        experiments
    ):

        print(
            "\n\n"
            + "#" * 80
        )

        print(
            f"EXPERIMENT "
            f"{i + 1}/{len(experiments)}"
        )

        print(
            "#" * 80
        )

        # 每个实验重新初始化随机种子
        seed_all(SEED)

        result = run_experiment(
            exp
        )

        result.append(
            result
        )

        save_summary(
            result
        )

    # Final summary

    print(
        "\n\n"
        + "=" * 80
    )

    print(
        "ALL EXPERIMENTS FINISHED"
    )

    print(
        "=" * 80
    )

    save_summary(
        result
    )


if __name__ == "__main__":

    main()