import os
import random
import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HGTConv


warnings.filterwarnings("ignore")

# Configuration
GRAPH_PATH = "result/hetero_graph.pt"
META_PATH = "result/meta.pkl"
SAVE_DIR = "checkpoints"
BEST_MODEL = os.path.join(
    SAVE_DIR,
    "best_rt_dhgnn_hgt.pt"
)

BATCH_SIZE = 4096

NUM_WORKERS = 6
PREFETCH_FACTOR = 4

HIDDEN_DIM = 128
HEADS = 4
NUM_LAYERS = 2

LR = 1e-3
WEIGHT_DECAY = 1e-4

EPOCHS = 50
PATIENCE = 10

SEED = 42

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

# Reproducibility
def seed_all(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


seed_all(SEED)


print("=" * 70)
print("RT-HG-HGT")
print("=" * 70)

print("Device:", device)

# Load graph
print("\nLoading graph...")

data = torch.load(
    GRAPH_PATH,
    weights_only=False
)

with open(
    META_PATH,
    "rb"
) as f:

    meta = pickle.load(f)


NUM_CLASSES = int(
    meta["num_classes"]
)

NUM_ACTIVITIES = NUM_CLASSES

NUM_HOPS = int(
    data["event"].hop_id.max().item()
) + 1

NUM_ROLES = int(
    data["role"].num_nodes
)


print("\nGraph metadata")
print("-" * 70)

print("Num event nodes :", data["event"].num_nodes)
print("Num role nodes  :", NUM_ROLES)
print("Num activities  :", NUM_ACTIVITIES)
print("Num hop IDs     :", NUM_HOPS)

print("\nEdge types:")

for edge_type in data.edge_types:

    print(
        f"{edge_type}: "
        f"{data[edge_type].edge_index.shape[1]}"
    )

# Check expected edge types
EXPECTED_EDGE_TYPES = {

    (
        "event",
        "historical_follow",
        "event"
    ),

    (
        "role",
        "executes",
        "event"
    ),

    (
        "event",
        "rev_executes",
        "role"
    ),

    (
        "role",
        "role_transition",
        "role"
    ),

    (
        "role",
        "role_transition_rev",
        "role"
    )
}


ACTUAL_EDGE_TYPES = set(
    data.edge_types
)


if ACTUAL_EDGE_TYPES != EXPECTED_EDGE_TYPES:

    print("\nERROR: Graph edge types do not match!")

    print(
        "Expected:",
        EXPECTED_EDGE_TYPES
    )

    print(
        "Actual:",
        ACTUAL_EDGE_TYPES
    )

    raise RuntimeError(
        "Graph structure mismatch."
    )


print(
    "\n✔ Graph edge types verified"
)

# Masks
train_mask = data[
    "event"
].train_mask

val_mask = data[
    "event"
].val_mask

test_mask = data[
    "event"
].test_mask


train_idx = torch.where(
    train_mask
)[0]

val_idx = torch.where(
    val_mask
)[0]

test_idx = torch.where(
    test_mask
)[0]


print("\nPrediction samples")
print("-" * 70)

print(
    "Train:",
    train_idx.numel()
)

print(
    "Val  :",
    val_idx.numel()
)

print(
    "Test :",
    test_idx.numel()
)

# Sanity check
assert (
    train_mask & val_mask
).sum().item() == 0

assert (
    train_mask & test_mask
).sum().item() == 0

assert (
    val_mask & test_mask
).sum().item() == 0


print(
    "\n✔ Train / Val / Test masks are mutually exclusive"
)

# Verify last-event labels
valid_labels = data[
    "event"
].y >= 0

print(
    "\nValid prediction labels:",
    valid_labels.sum().item()
)

# Neighbor Loader
def build_loader(
    idx,
    shuffle=False
):

    return NeighborLoader(

        data=data,

        input_nodes=(
            "event",
            idx
        ),

        num_neighbors={
            edge_type: [20, 10]
            for edge_type
            in data.edge_types
        },

        batch_size=BATCH_SIZE,

        shuffle=shuffle,

        num_workers=NUM_WORKERS,

        pin_memory=True,

        persistent_workers=(
            NUM_WORKERS > 0
        ),

        prefetch_factor=(
            PREFETCH_FACTOR
            if NUM_WORKERS > 0
            else None
        )
    )


print("\nBuilding NeighborLoaders...")

train_loader = build_loader(
    train_idx,
    shuffle=True
)

val_loader = build_loader(
    val_idx,
    shuffle=False
)

test_loader = build_loader(
    test_idx,
    shuffle=False
)

print(
    "✔ NeighborLoaders created"
)

# Model
class HGTEncoder(
    nn.Module
):

    def __init__(self):

        super().__init__()

        # Activity embedding
        self.act_emb = nn.Embedding(
            NUM_ACTIVITIES,
            64
        )

        # Historical role-transition/activity hop embedding
        self.hop_emb = nn.Embedding(
            NUM_HOPS,
            32
        )

        # Time interval
        self.delta = nn.Sequential(

            nn.Linear(
                1,
                32
            ),

            nn.ReLU()
        )

        # Role node feature
        self.role = nn.Sequential(

            nn.Linear(
                2,
                HIDDEN_DIM
            ),

            nn.ReLU()
        )

        # Event feature fusion
        self.fusion = nn.Sequential(

            nn.Linear(
                128,
                HIDDEN_DIM
            ),

            nn.ReLU()
        )
        # HGT layers
        self.convs = nn.ModuleList()

        for _ in range(
            NUM_LAYERS
        ):

            self.convs.append(

                HGTConv(

                    in_channels=HIDDEN_DIM,

                    out_channels=HIDDEN_DIM,

                    metadata=data.metadata(),

                    heads=HEADS
                )
            )

        # Type-specific normalization
        self.event_norms = nn.ModuleList([

            nn.LayerNorm(
                HIDDEN_DIM
            )

            for _ in range(
                NUM_LAYERS
            )
        ])

        self.role_norms = nn.ModuleList([

            nn.LayerNorm(
                HIDDEN_DIM
            )

            for _ in range(
                NUM_LAYERS
            )
        ])

        # Classifier
        self.cls = nn.Sequential(

            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.Dropout(
                0.1
            ),

            nn.Linear(
                HIDDEN_DIM,
                NUM_CLASSES
            )
        )

    # Forward
    def forward(
        self,
        batch
    ):

        # Feature encoding
        amp_enabled = (
            device.type == "cuda"
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=amp_enabled
        ):

            # Event activity
            act = self.act_emb(
                batch[
                    "event"
                ].x_int.squeeze(-1)
            )

            # Historical hop
            hop = self.hop_emb(
                batch[
                    "event"
                ].hop_id
            )

            # Delta time
            delta = self.delta(
                batch[
                    "event"
                ].x_float
            )

            # Event fusion

            x_event = torch.cat(
                [
                    act,
                    hop,
                    delta
                ],
                dim=-1
            )

            x_event = self.fusion(
                x_event
            )

            # Role feature
            x_role = self.role(
                batch[
                    "role"
                ].x
            )

        # HGT domain
        x_dict = {

            "event":
                x_event.float(),

            "role":
                x_role.float()
        }

        edge_index_dict = (
            batch.edge_index_dict
        )

        for layer_idx, (
            conv,
            event_norm,
            role_norm
        ) in enumerate(
            zip(
                self.convs,
                self.event_norms,
                self.role_norms
            )
        ):

            residual = {

                key:
                    value

                for key, value
                in x_dict.items()
            }

            # HGTConv FP32 SAFE DOMAIN
            with torch.amp.autocast(
                device_type="cuda",
                enabled=False
            ):

                x_dict = conv(
                    x_dict,
                    edge_index_dict
                )

            # Residual + type-specific normalization
            x_dict["event"] = event_norm(
                x_dict["event"]
                + residual["event"]
            )

            x_dict["role"] = role_norm(
                x_dict["role"]
                + residual["role"]
            )

            x_dict["event"] = F.relu(
                x_dict["event"]
            )

            x_dict["role"] = F.relu(
                x_dict["role"]
            )

        # Classifier
        logits = self.cls(
            x_dict["event"]
        )

        return logits

# Create model
model = HGTEncoder().to(device)

print(
    "\nModel parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)

# Optimizer
optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LR,

    weight_decay=WEIGHT_DECAY
)


scheduler = (
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )
)

# AMP scaler
if device.type == "cuda":

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:
    scaler = None

# Train one epoch
def train_one_epoch():

    model.train()

    total_loss = 0.0

    total_samples = 0

    amp_enabled = (
        device.type == "cuda"
    )

    for batch in train_loader:

        batch = batch.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=amp_enabled
        ):

            logits = model(
                batch
            )

            n = (
                batch[
                    "event"
                ].batch_size
            )

            logits_seed = (
                logits[:n]
            )

            y_seed = (
                batch[
                    "event"
                ].y[:n]
            )

            mask = (
                y_seed >= 0
            )

            if mask.sum().item() == 0:
                continue

            loss = F.cross_entropy(

                logits_seed[mask],

                y_seed[mask],

                label_smoothing=0.05
            )

        # Backward
        if scaler is not None:

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                2.0
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                2.0
            )

            optimizer.step()

        # Loss statistics
        batch_samples = (
            mask.sum().item()
        )

        total_loss += (
            loss.item()
            * batch_samples
        )

        total_samples += (
            batch_samples
        )

    scheduler.step()

    if total_samples == 0:

        return 0.0

    return (
        total_loss
        / total_samples
    )

# Evaluation
@torch.no_grad()
def evaluate(
    loader
):

    model.eval()

    y_true = []
    y_pred = []

    amp_enabled = (
        device.type == "cuda"
    )

    for batch in loader:

        batch = batch.to(
            device,
            non_blocking=True
        )

        # Forward
        with torch.amp.autocast(
            device_type="cuda",
            enabled=amp_enabled
        ):

            logits = model(
                batch
            )

        n = (
            batch[
                "event"
            ].batch_size
        )

        logits = logits[:n]

        y = (
            batch[
                "event"
            ].y[:n]
        )

        mask = (
            y >= 0
        )

        if mask.sum().item() == 0:
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

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return acc, f1

# Training
print("\n")
print("=" * 70)
print("Training")
print("=" * 70)

best_f1 = -1.0
best_epoch = -1

patience_count = 0

for epoch in range(
    EPOCHS
):

    loss = train_one_epoch()

    val_acc, val_f1 = evaluate(
        val_loader
    )

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:02d} | "
        f"Loss {loss:.6f} | "
        f"Val Acc {val_acc:.6f} | "
        f"Val F1 {val_f1:.6f} | "
        f"LR {current_lr:.7f}"
    )

    # Save best model according to Macro-F1
    if val_f1 > best_f1:

        best_f1 = val_f1

        best_epoch = epoch

        patience_count = 0

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_acc":
                    val_acc,

                "val_f1":
                    val_f1,

                "config": {
                    "hidden_dim": HIDDEN_DIM,
                    "heads": HEADS,
                    "num_layers": NUM_LAYERS,
                    "lr": LR,
                    "weight_decay":
                        WEIGHT_DECAY
                }
            },
            BEST_MODEL
        )

        print(
            "✔ Saved Best Model"
        )

    else:

        patience_count += 1

        if patience_count >= PATIENCE:

            print(
                "Early Stop"
            )

            break

# Load best model
print("\n")
print("=" * 70)
print("Loading Best Model")
print("=" * 70)

checkpoint = torch.load(
    BEST_MODEL,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


print(
    "Best Epoch:",
    checkpoint["epoch"]
)

print(
    "Best Val Acc:",
    checkpoint["val_acc"]
)

print(
    "Best Val F1:",
    checkpoint["val_f1"]
)
# Final Test
print("\n")
print("=" * 70)
print("Final Test")
print("=" * 70)

test_acc, test_f1 = evaluate(
    test_loader
)


print("\nFINAL RESULT")

print(
    "ACC:",
    test_acc
)

print(
    "F1 :",
    test_f1
)

print(
    "\nBest checkpoint:",
    BEST_MODEL
)

print("=" * 70)