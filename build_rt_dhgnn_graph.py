import os
import pickle
import random
import warnings

import numpy as np
import pandas as pd
import torch

from torch_geometric.data import HeteroData

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

warnings.filterwarnings("ignore")



# Config
CSV_PATH = "data/BPIC2017_role_enhanced.csv"

OUTPUT_DIR = "result"
OUTPUT_GRAPH = os.path.join(
    OUTPUT_DIR,
    "hetero_graph.pt"
)
OUTPUT_META = os.path.join(
    OUTPUT_DIR,
    "meta.pkl"
)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

SEED = 42

# Special hop token.
# 0 is reserved for UNK.
UNK_HOP_ID = 0


# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# Utility
def print_separator(title=None):
    print("\n" + "=" * 70)

    if title is not None:
        print(title)
        print("=" * 70)

# Load Dataset
print_separator("Loading")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

required_columns = [
    "case_id",
    "timestamp",
    "activity",
    "role_label"
]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + str(missing_columns)
    )


print("Raw Events :", len(df))
print("Raw Cases  :", df["case_id"].nunique())
print("Raw Roles  :", df["role_label"].nunique())
print("Activities :", df["activity"].nunique())

# Timestamp
print_separator("Processing Timestamp")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="mixed",
    utc=True,
    errors="raise"
)

# Sort events inside each case.
df = (
    df.sort_values(
        ["case_id", "timestamp"],
        kind="mergesort"
    )
    .reset_index(drop=True)
)

print("Timestamp conversion completed.")


# Encode Activity / Role
print_separator("Encoding Activity and Role")

activity_encoder = LabelEncoder()
role_encoder = LabelEncoder()

df["activity_id"] = (
    activity_encoder.fit_transform(
        df["activity"].astype(str)
    )
)

df["role_id"] = (
    role_encoder.fit_transform(
        df["role_label"].astype(str)
    )
)

num_activities = len(
    activity_encoder.classes_
)

num_roles = len(
    role_encoder.classes_
)

print("Activities :", num_activities)
print("Roles      :", num_roles)

print(
    "Activity vocabulary:",
    list(activity_encoder.classes_)
)

print(
    "Role vocabulary:",
    list(role_encoder.classes_)
)

# Case-level Split
print_separator("Case-level Train / Val / Test Split")

cases = np.array(
    df["case_id"].unique()
)

rng = np.random.default_rng(SEED)
rng.shuffle(cases)

n_cases = len(cases)

n_train = int(
    n_cases * TRAIN_RATIO
)

n_val = int(
    n_cases * VAL_RATIO
)

train_cases = set(
    cases[:n_train]
)

val_cases = set(
    cases[
        n_train:
        n_train + n_val
    ]
)

test_cases = set(
    cases[
        n_train + n_val:
    ]
)

print("Total Cases :", n_cases)
print("Train Cases :", len(train_cases))
print("Val Cases   :", len(val_cases))
print("Test Cases  :", len(test_cases))

assert (
    len(train_cases)
    + len(val_cases)
    + len(test_cases)
    == n_cases
)

# Split ID
def get_split(case_id):

    if case_id in train_cases:
        return "train"

    if case_id in val_cases:
        return "val"

    if case_id in test_cases:
        return "test"

    raise ValueError(
        f"Unknown case_id: {case_id}"
    )


df["split"] = df["case_id"].map(
    get_split
)


print("\nEvent split:")

print(
    df["split"]
    .value_counts()
    .sort_index()
)

# Delta Time
print_separator(
    "Computing Delta Time"
)

delta_t = np.zeros(
    len(df),
    dtype=np.float64
)

for case_id, idx in df.groupby(
    "case_id",
    sort=False
).groups.items():

    idx = np.asarray(
        list(idx),
        dtype=np.int64
    )

    ts = df.loc[
        idx,
        "timestamp"
    ]

    diff = (
        ts.diff()
        .dt.total_seconds()
        .fillna(0.0)
        .to_numpy()
    )

    delta_t[idx] = diff


df["delta_t"] = delta_t

# Train-only Delta Time Scaling
print_separator(
    "Train-only Delta Time Scaling"
)

delta_scaler = StandardScaler()

train_event_mask_for_scaler = (
    df["split"] == "train"
)

delta_scaler.fit(
    df.loc[
        train_event_mask_for_scaler,
        ["delta_t"]
    ]
)

df["delta_t_norm"] = (
    delta_scaler.transform(
        df[["delta_t"]]
    )
    .astype(np.float32)
)


print(
    "Train delta_t mean :",
    float(delta_scaler.mean_[0])
)

print(
    "Train delta_t std  :",
    float(delta_scaler.scale_[0])
)

# Containers
n_events = len(df)

# Target
y = np.full(
    n_events,
    -1,
    dtype=np.int64
)

# Masks
train_mask = np.zeros(
    n_events,
    dtype=bool
)

val_mask = np.zeros(
    n_events,
    dtype=bool
)

test_mask = np.zeros(
    n_events,
    dtype=bool
)

# Hop IDs
hop_ids = np.zeros(
    n_events,
    dtype=np.int64
)

# Historical control-flow edges
historical_src = []
historical_dst = []

# Execution edges
exec_src = []
exec_dst = []

# Train-only role transition counts
rt_counter = {}

# Build historical control flow and labels
print_separator(
    "Building Historical Control Flow"
)

for case_id, trace in df.groupby(
    "case_id",
    sort=False
):

    idx = trace.index.to_numpy(
        dtype=np.int64
    )

    acts = trace[
        "activity_id"
    ].to_numpy(
        dtype=np.int64
    )

    roles = trace[
        "role_id"
    ].to_numpy(
        dtype=np.int64
    )

    split = get_split(case_id)

    # Execution relation
    exec_src.extend(
        roles.tolist()
    )

    exec_dst.extend(
        idx.tolist()
    )

    # Event sequence
    for i in range(len(idx)):

        cur_node = int(
            idx[i]
        )
        if i > 0:
            prev_node = int(
                idx[i - 1]
            )

            historical_src.append(
                prev_node
            )

            historical_dst.append(
                cur_node
            )

        if i == len(idx) - 1:
            y[cur_node] = -1

            continue

        y[cur_node] = int(
            acts[i + 1]
        )

        # Prediction mask
        if split == "train":

            train_mask[cur_node] = True

        elif split == "val":

            val_mask[cur_node] = True

        elif split == "test":

            test_mask[cur_node] = True

        else:

            raise ValueError(
                f"Unknown split: {split}"
            )

        # Train-only role transition statistics
        if (
                split == "train"
                and roles[i] != roles[i + 1]
        ):
            r1 = int(
                roles[i]
            )

            r2 = int(
                roles[i + 1]
            )

            rt_counter[
                (r1, r2)
            ] = (
                    rt_counter.get(
                        (r1, r2),
                        0
                    ) + 1
            )

        # Train-only role transition statistics
        if (
            split == "train"
            and roles[i] != roles[i + 1]
        ):

            r1 = int(
                roles[i]
            )

            r2 = int(
                roles[i + 1]
            )

            rt_counter[
                (r1, r2)
            ] = (
                rt_counter.get(
                    (r1, r2),
                    0
                ) + 1
            )

# Build Train-only Hop Vocabulary
print_separator(
    "Building Train-only Hop Vocabulary"
)

hop_encoder = {
    "<UNK>": UNK_HOP_ID
}

hop_counter = 1


def make_hop_pattern(
    prev_role,
    current_role,
    current_activity
):

    if prev_role is None:

        return (
            "START->"
            + str(current_role)
            + "|"
            + str(current_activity)
        )

    return (
        str(prev_role)
        + "->"
        + str(current_role)
        + "|"
        + str(current_activity)
    )

for case_id, trace in df.groupby(
    "case_id",
    sort=False
):

    if case_id not in train_cases:
        continue

    idx = trace.index.to_numpy(
        dtype=np.int64
    )

    acts = trace[
        "activity_id"
    ].to_numpy(
        dtype=np.int64
    )

    roles = trace[
        "role_id"
    ].to_numpy(
        dtype=np.int64
    )

    for i in range(len(idx)):

        if i == 0:
            prev_role = None
        else:
            prev_role = int(
                roles[i - 1]
            )

        current_role = int(
            roles[i]
        )

        current_activity = int(
            acts[i]
        )

        pattern = make_hop_pattern(
            prev_role,
            current_role,
            current_activity
        )

        if pattern not in hop_encoder:

            hop_encoder[
                pattern
            ] = hop_counter

            hop_counter += 1


print(
    "Train Hop Vocabulary :",
    len(hop_encoder)
)

print(
    "UNK Hop ID           :",
    UNK_HOP_ID
)

# Assign Hop IDs to ALL events
print_separator(
    "Assigning Historical Hop IDs"
)

unknown_hop_count = {
    "train": 0,
    "val": 0,
    "test": 0
}

known_hop_count = {
    "train": 0,
    "val": 0,
    "test": 0
}


for case_id, trace in df.groupby(
    "case_id",
    sort=False
):

    idx = trace.index.to_numpy(
        dtype=np.int64
    )

    acts = trace[
        "activity_id"
    ].to_numpy(
        dtype=np.int64
    )

    roles = trace[
        "role_id"
    ].to_numpy(
        dtype=np.int64
    )

    split = get_split(case_id)

    for i in range(len(idx)):

        if i == 0:
            prev_role = None
        else:
            prev_role = int(
                roles[i - 1]
            )

        current_role = int(
            roles[i]
        )

        current_activity = int(
            acts[i]
        )

        pattern = make_hop_pattern(
            prev_role,
            current_role,
            current_activity
        )

        if pattern in hop_encoder:

            hop_ids[
                idx[i]
            ] = hop_encoder[
                pattern
            ]

            known_hop_count[
                split
            ] += 1

        else:

            hop_ids[
                idx[i]
            ] = UNK_HOP_ID

            unknown_hop_count[
                split
            ] += 1


print(
    "\nKnown Hop Counts:"
)

print(
    known_hop_count
)

print(
    "\nUnknown Hop Counts:"
)

print(
    unknown_hop_count
)

# Role Transition Graph
print_separator(
    "Building Train-only Role Transition Graph"
)

rt_src = []
rt_dst = []
rt_weight = []

# Calculate outgoing counts
out_sum = {}

for (
    r1,
    r2
), count in rt_counter.items():

    out_sum[r1] = (
        out_sum.get(
            r1,
            0
        )
        + count
    )

# Conditional probability
for (
    r1,
    r2
), count in sorted(
    rt_counter.items()
):

    if out_sum[r1] <= 0:
        continue

    probability = (
        count
        / out_sum[r1]
    )

    rt_src.append(
        r1
    )

    rt_dst.append(
        r2
    )

    rt_weight.append(
        probability
    )


print(
    "Role Transition Types :",
    len(rt_src)
)

# Verify Transition Probabilities
print_separator(
    "Checking Role Transition Probabilities"
)

prob_sum = {}

for s, w in zip(
    rt_src,
    rt_weight
):

    prob_sum[s] = (
        prob_sum.get(
            s,
            0.0
        )
        + w
    )

for role_id in sorted(
    prob_sum.keys()
):

    print(
        f"Role {role_id}: "
        f"outgoing probability sum = "
        f"{prob_sum[role_id]:.6f}"
    )

# Role Features
print_separator(
    "Building Role Features"
)

out_degree = np.zeros(
    num_roles,
    dtype=np.float32
)

in_degree = np.zeros(
    num_roles,
    dtype=np.float32
)

for s, d, w in zip(
    rt_src,
    rt_dst,
    rt_weight
):

    out_degree[s] += float(w)

    in_degree[d] += float(w)


role_x = np.column_stack([
    out_degree,
    in_degree
]).astype(
    np.float32
)


print(
    "Role feature shape:",
    role_x.shape
)

# Event Features
print_separator(
    "Building Event Features"
)

event_x_int = (
    df["activity_id"]
    .to_numpy(
        dtype=np.int64
    )
    .reshape(-1, 1)
)

event_x_float = (
    df["delta_t_norm"]
    .to_numpy(
        dtype=np.float32
    )
    .reshape(-1, 1)
)

# Sanity Checks
print_separator(
    "Running Leakage Sanity Checks"
)

assert np.all(
    y[train_mask]
    >= 0
)

assert np.all(
    y[val_mask]
    >= 0
)

assert np.all(
    y[test_mask]
    >= 0
)

print(
    "Train prediction samples:",
    int(train_mask.sum())
)

print(
    "Val prediction samples  :",
    int(val_mask.sum())
)

print(
    "Test prediction samples :",
    int(test_mask.sum())
)

last_event_indices = []

for case_id, trace in df.groupby(
    "case_id",
    sort=False
):

    last_idx = int(
        trace.index[-1]
    )

    last_event_indices.append(
        last_idx
    )


last_event_indices = np.asarray(
    last_event_indices,
    dtype=np.int64
)

assert np.all(
    y[last_event_indices] == -1
)

assert not np.any(
    train_mask[last_event_indices]
)

assert not np.any(
    val_mask[last_event_indices]
)

assert not np.any(
    test_mask[last_event_indices]
)

print(
    "Last-event label check: PASS"
)

expected_historical_edges = (
    n_events
    - n_cases
)

actual_historical_edges = len(
    historical_src
)

print(
    "Expected historical edges:",
    expected_historical_edges
)

print(
    "Actual historical edges  :",
    actual_historical_edges
)

assert (
    expected_historical_edges
    == actual_historical_edges
)

# HeteroData
print_separator(
    "Building HeteroData"
)

data = HeteroData()

# Event Nodes
data["event"].num_nodes = (
    n_events
)

data["event"].x_int = torch.tensor(
    event_x_int,
    dtype=torch.long
)

data["event"].x_float = torch.tensor(
    event_x_float,
    dtype=torch.float
)

data["event"].hop_id = torch.tensor(
    hop_ids,
    dtype=torch.long
)

data["event"].y = torch.tensor(
    y,
    dtype=torch.long
)

data["event"].train_mask = torch.tensor(
    train_mask,
    dtype=torch.bool
)

data["event"].val_mask = torch.tensor(
    val_mask,
    dtype=torch.bool
)

data["event"].test_mask = torch.tensor(
    test_mask,
    dtype=torch.bool
)

# Role Nodes
data["role"].num_nodes = (
    num_roles
)

data["role"].x = torch.tensor(
    role_x,
    dtype=torch.float
)

# Historical Control Flow
historical_edge_index = torch.tensor(
    [
        historical_src,
        historical_dst
    ],
    dtype=torch.long
)

data[
    "event",
    "historical_follow",
    "event"
].edge_index = (
    historical_edge_index
)

# Role -> Event
# executes
exec_edge_index = torch.tensor(
    [
        exec_src,
        exec_dst
    ],
    dtype=torch.long
)

data[
    "role",
    "executes",
    "event"
].edge_index = (
    exec_edge_index
)

# Event -> Role
# rev_executes

data[
    "event",
    "rev_executes",
    "role"
].edge_index = torch.tensor(
    [
        exec_dst,
        exec_src
    ],
    dtype=torch.long
)

# Role Transition
data[
    "role",
    "role_transition",
    "role"
].edge_index = torch.tensor(
    [
        rt_src,
        rt_dst
    ],
    dtype=torch.long
)

data[
    "role",
    "role_transition",
    "role"
].edge_weight = torch.tensor(
    rt_weight,
    dtype=torch.float
)

# Role Transition Reverse
data[
    "role",
    "role_transition_rev",
    "role"
].edge_index = torch.tensor(
    [
        rt_dst,
        rt_src
    ],
    dtype=torch.long
)

data[
    "role",
    "role_transition_rev",
    "role"
].edge_weight = torch.tensor(
    rt_weight,
    dtype=torch.float
)

# Metadata
meta = {
    # Encoders
    "activity_encoder":
        activity_encoder,

    "role_encoder":
        role_encoder,

    "activity_vocab":
        list(
            activity_encoder.classes_
        ),

    "role_vocab":
        list(
            role_encoder.classes_
        ),

    "num_classes":
        num_activities,

    "num_activities":
        num_activities,

    "num_roles":
        num_roles,

    # Hop
    "hop_encoder":
        hop_encoder,

    "num_hops":
        len(hop_encoder),

    "unk_hop_id":
        UNK_HOP_ID,

    "hop_definition":
        "(previous_role -> current_role, current_activity)",

    # Split
    "seed":
        SEED,

    "train_ratio":
        TRAIN_RATIO,

    "val_ratio":
        VAL_RATIO,

    "test_ratio":
        TEST_RATIO,

    "num_cases":
        n_cases,

    "train_cases":
        list(train_cases),

    "val_cases":
        list(val_cases),

    "test_cases":
        list(test_cases),

    # Scaler
    "delta_scaler":
        delta_scaler,

    # Role transition
    "role_transition_count":
        rt_counter,

    "role_transition_probability":
        {
            (int(s), int(d)): float(w)
            for s, d, w in zip(
                rt_src,
                rt_dst,
                rt_weight
            )
        },

    "role_transition_definition":
        "P(r_j | r_i), estimated from training cases only",

    # Graph definition
    "edge_types": [
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
    ],

    # Removed leakage-prone relation
    "removed_edge_types": [
        (
            "event",
            "rev_directly_follow",
            "event"
        )
    ],

    "causal_constraint":
        "Prediction of a_{t+1} at e_t may only use information available at t.",

    "target_definition":
        "y[e_t] = activity(e_{t+1})",

    "historical_follow_definition":
        "e_{t-1} -> e_t",

    # Dataset statistics
    "num_events":
        n_events,

    "num_prediction_samples":
        int(
            train_mask.sum()
            + val_mask.sum()
            + test_mask.sum()
        ),

    "num_train_samples":
        int(
            train_mask.sum()
        ),

    "num_val_samples":
        int(
            val_mask.sum()
        ),

    "num_test_samples":
        int(
            test_mask.sum()
        )
}

# Save
print_separator(
    "Saving Graph"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

torch.save(
    data,
    OUTPUT_GRAPH
)

with open(
    OUTPUT_META,
    "wb"
) as f:

    pickle.dump(
        meta,
        f
    )

# Final Summary
print_separator(
    "FINAL GRAPH SUMMARY"
)

print(
    "Event Nodes :",
    data["event"].num_nodes
)

print(
    "Role Nodes  :",
    data["role"].num_nodes
)

print(
    "\nEdge Types:"
)

for edge_type in data.edge_types:

    edge_count = (
        data[edge_type]
        .edge_index
        .shape[1]
    )

    print(
        f"{edge_type}: {edge_count}"
    )


print(
    "\nHop Patterns :",
    len(hop_encoder)
)

print(
    "Train Hop Patterns:",
    len(hop_encoder) - 1
)

print(
    "UNK Hop ID:",
    UNK_HOP_ID
)

print(
    "\nPrediction Samples:"
)

print(
    "Train:",
    int(train_mask.sum())
)

print(
    "Val  :",
    int(val_mask.sum())
)

print(
    "Test :",
    int(test_mask.sum())
)

print(
    "\nRole Transition Types:",
    len(rt_src)
)

print(
    "\nLeakage Protection:"
)

print(
    "  [PASS] Case-level split"
)

print(
    "  [PASS] Delta scaler fitted on TRAIN only"
)

print(
    "  [PASS] Role transition estimated on TRAIN only"
)

print(
    "  [PASS] Hop vocabulary built on TRAIN only"
)

print(
    "  [PASS] Historical hop uses previous role"
)

print(
    "  [PASS] No future event -> current event edge"
)

print(
    "  [PASS] role_transition_rev retained"
)

print(
    "  [PASS] Last event excluded from prediction"
)

print(
    "\nSaved:"
)

print(
    "  Graph:",
    OUTPUT_GRAPH
)

print(
    "  Meta :",
    OUTPUT_META
)

print_separator(
    "Finished"
)

