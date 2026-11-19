# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

from scipy.sparse import hstack

from scipy.stats import wilcoxon

import matplotlib.pyplot as plt

# Load
print("=" * 60)
print("Loading...")
print("=" * 60)

df = pd.read_csv(
    "data/BPIC2017_org_enhanced.csv"
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="mixed",
    utc=True
)

df = df.sort_values(
    ["case_id", "timestamp"]
)

print("Events :", len(df))
print("Cases  :", df["case_id"].nunique())

# Build Samples
records = []

for cid, trace in df.groupby("case_id"):

    acts = trace["activity"].tolist()
    roles = trace["org_label"].tolist()

    for i in range(len(trace) - 1):

        records.append([
            cid,
            acts[i],
            roles[i] + "->" + roles[i + 1],
            acts[i + 1]
        ])

data = pd.DataFrame(
    records,
    columns=[
        "case_id",
        "activity",
        "role_transition",
        "next_activity"
    ]
)

print("Samples :", len(data))

# Case Split
cases = data["case_id"].unique()

np.random.seed(42)
np.random.shuffle(cases)

split = int(len(cases) * 0.7)

train_cases = set(cases[:split])
test_cases = set(cases[split:])

train = data[
    data["case_id"].isin(train_cases)
]

test = data[
    data["case_id"].isin(test_cases)
]

print("Train :", len(train))
print("Test  :", len(test))

# Encoding
act_ohe = OneHotEncoder(
    sparse_output=True,
    handle_unknown="ignore"
)

rt_ohe = OneHotEncoder(
    sparse_output=True,
    handle_unknown="ignore"
)

y_enc = LabelEncoder()

A_train = act_ohe.fit_transform(
    train[["activity"]]
)

A_test = act_ohe.transform(
    test[["activity"]]
)

RT_train = rt_ohe.fit_transform(
    train[["role_transition"]]
)

RT_test = rt_ohe.transform(
    test[["role_transition"]]
)

Y_train = y_enc.fit_transform(
    train["next_activity"]
)

Y_test = y_enc.transform(
    test["next_activity"]
)

print(
    "Activity Features :",
    A_train.shape
)

print(
    "RoleTransition Features :",
    RT_train.shape
)

# Real Model
print("\nTraining Real Model...")

X_train_real = hstack([
    A_train,
    RT_train
])

X_test_real = hstack([
    A_test,
    RT_test
])

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train_real,
    Y_train
)

pred = rf.predict(
    X_test_real
)

acc_real = accuracy_score(
    Y_test,
    pred
)

f1_real = f1_score(
    Y_test,
    pred,
    average="macro"
)

print(
    "Real ACC:",
    round(acc_real, 4)
)

print(
    "Real F1 :",
    round(f1_real, 4)
)

# Shuffle Test
acc_real_list = []
acc_shuffle_list = []

f1_real_list = []
f1_shuffle_list = []

for seed in range(10):

    print(
        f"\nShuffle Run {seed+1}/10"
    )

    acc_real_list.append(
        acc_real
    )

    f1_real_list.append(
        f1_real
    )

    idx = np.random.RandomState(
        seed
    ).permutation(
        RT_train.shape[0]
    )

    RT_shuffle = RT_train[idx]

    X_train_shuffle = hstack([
        A_train,
        RT_shuffle
    ])

    rf_s = RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        n_jobs=-1
    )

    rf_s.fit(
        X_train_shuffle,
        Y_train
    )

    pred_s = rf_s.predict(
        X_test_real
    )

    acc_s = accuracy_score(
        Y_test,
        pred_s
    )

    f1_s = f1_score(
        Y_test,
        pred_s,
        average="macro"
    )

    acc_shuffle_list.append(
        acc_s
    )

    f1_shuffle_list.append(
        f1_s
    )

# Wilcoxon
print("\n")
print("=" * 60)
print("Wilcoxon Test")
print("=" * 60)

stat, p = wilcoxon(
    acc_real_list,
    acc_shuffle_list,
    alternative="greater"
)

print("p-value =", p)

# Result Table
result = pd.DataFrame({

    "Model": [
        "Activity+RT",
        "Activity+ShuffleRT"
    ],

    "Accuracy": [
        np.mean(acc_real_list),
        np.mean(acc_shuffle_list)
    ],

    "MacroF1": [
        np.mean(f1_real_list),
        np.mean(f1_shuffle_list)
    ]
})

print("\n")
print(result)

result.to_csv(
    "Table_ShuffleTest.csv",
    index=False
)

# ACC
plt.figure(figsize=(6,5))

plt.bar(
    result["Model"],
    result["Accuracy"]
)

plt.ylabel("Accuracy")

plt.title(
    "Shuffle Robustness Test"
)

plt.tight_layout()

plt.savefig(
    "result/Fig_Shuffle_ACC.png",
    dpi=300
)

# F1
plt.figure(figsize=(6,5))

plt.bar(
    result["Model"],
    result["MacroF1"]
)

plt.ylabel("Macro-F1")

plt.title(
    "Shuffle Robustness Test"
)

plt.tight_layout()

plt.savefig(
    "result/Fig_Shuffle_F1.png",
    dpi=300
)

# Summary
print("\n")
print("=" * 60)
print("Summary")
print("=" * 60)

drop_acc = (
    acc_real
    - np.mean(acc_shuffle_list)
)

print(
    "Accuracy Drop:",
    round(drop_acc, 4)
)

print(
    "Relative Drop:",
    round(
        drop_acc / acc_real * 100,
        2
    ),
    "%"
)

print("\nSaved:")
print("Table_ShuffleTest.csv")
print("Fig_Shuffle_ACC.png")
print("Fig_Shuffle_F1.png")
