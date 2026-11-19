# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score
)

from scipy.stats import wilcoxon

import matplotlib.pyplot as plt

# Load
print("=" * 60)
print("Loading...")

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
).reset_index(drop=True)

print("Events :", len(df))
print("Cases  :", df["case_id"].nunique())
print("Roles  :", df["org_label"].nunique())

# Build Role Transition
print("\nBuilding Role Transition...")

activity = []
role = []
role_transition = []
next_activity = []

for _, trace in df.groupby("case_id"):

    acts = trace["activity"].tolist()
    roles = trace["org_label"].tolist()

    for i in range(len(trace)-1):

        if roles[i] == roles[i+1]:
            continue

        activity.append(acts[i])

        role.append(roles[i])

        role_transition.append(
            roles[i] + "->" + roles[i+1]
        )

        next_activity.append(
            acts[i+1]
        )

data = pd.DataFrame({
    "activity": activity,
    "role": role,
    "role_transition": role_transition,
    "next_activity": next_activity
})

print("Samples:", len(data))

# Encoding
act_enc = LabelEncoder()
role_enc = LabelEncoder()
y_enc = LabelEncoder()

A = act_enc.fit_transform(
    data["activity"]
)

R = role_enc.fit_transform(
    data["role"]
)

Y = y_enc.fit_transform(
    data["next_activity"]
)

# RoleTransition OneHot
try:

    rt_ohe = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

except:

    rt_ohe = OneHotEncoder(
        sparse=False,
        handle_unknown="ignore"
    )

RT = rt_ohe.fit_transform(
    data[["role_transition"]]
)

print(
    "\nRoleTransition Types:",
    RT.shape[1]
)

# Feature Sets
X_activity = A.reshape(-1,1)

X_role = R.reshape(-1,1)

X_rt = RT

X_activity_role = np.column_stack([
    A,
    R
])

X_activity_rt = np.hstack([
    A.reshape(-1,1),
    RT
])

# Cross Validation
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)

# RF
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

# Evaluate
def evaluate(X,name):

    print("\n",name)

    acc = cross_val_score(
        rf,
        X,
        Y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    f1 = cross_val_score(
        rf,
        X,
        Y,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    print(
        "ACC:",
        round(acc.mean(),4)
    )

    print(
        "F1 :",
        round(f1.mean(),4)
    )

    return acc,f1

# Run
acc_act,f1_act = evaluate(
    X_activity,
    "Activity"
)

acc_role,f1_role = evaluate(
    X_role,
    "Role"
)

acc_rt,f1_rt = evaluate(
    X_rt,
    "RoleTransition"
)

acc_ar,f1_ar = evaluate(
    X_activity_role,
    "Activity+Role"
)

acc_art,f1_art = evaluate(
    X_activity_rt,
    "Activity+RoleTransition"
)

# Wilcoxon
print("\n")
print("="*60)
print("Wilcoxon Test")
print("="*60)

stat,p = wilcoxon(
    acc_act,
    acc_art,
    alternative="less"
)

print(
    "Activity vs Activity+RT"
)

print(
    "p-value:",
    p
)

# effect_size = (
#     acc_art.mean()
#     -
#     acc_act.mean()
# ) / acc_act.std()

pooled_std =np.sqrt(
    (
      acc_act.var()
      +
      acc_art.var()
    ) / 2
)

cohen_d =(
    acc_art.mean()
    -
    acc_act.mean()
) / pooled_std

print(
    "Effect Size:",
    round(cohen_d ,4)
)

# Result Table
result = pd.DataFrame({

    "Model":[
        "Activity",
        "Role",
        "RoleTransition",
        "Activity+Role",
        "Activity+RoleTransition"
    ],

    "Accuracy":[
        acc_act.mean(),
        acc_role.mean(),
        acc_rt.mean(),
        acc_ar.mean(),
        acc_art.mean()
    ],

    "MacroF1":[
        f1_act.mean(),
        f1_role.mean(),
        f1_rt.mean(),
        f1_ar.mean(),
        f1_art.mean()
    ]
})

print("\n")
print(result)

result.to_csv(
    "Table_IncrementalGain.csv",
    index=False
)

# Fig5 Accuracy
plt.figure(figsize=(8,5))

plt.bar(
    result["Model"],
    result["Accuracy"]
)

plt.xticks(rotation=20)

plt.ylabel("Accuracy")

plt.title(
    "Incremental Information Gain"
)

plt.tight_layout()

plt.savefig(
    "result/Fig5_Incremental_Accuracy.png",
    dpi=300
)

# Fig6 F1
plt.figure(figsize=(8,5))

plt.bar(
    result["Model"],
    result["MacroF1"]
)

plt.xticks(rotation=20)

plt.ylabel("Macro F1")

plt.title(
    "Incremental Information Gain"
)

plt.tight_layout()

plt.savefig(
    "result/Fig6_Incremental_F1.png",
    dpi=300
)

# Gain
gain = (
    acc_art.mean()
    -
    acc_act.mean()
)

print("\n")
print("="*60)
print("Incremental Gain")
print("="*60)

print(
    "Accuracy Gain:",
    round(gain,4)
)

print(
    "Relative Gain:",
    round(
        gain /
        acc_act.mean()
        *100,
        2
    ),
    "%"
)

print("\nSaved:")
print("Table_IncrementalGain.csv")
print("Fig5_Incremental_Accuracy.png")
print("Fig6_Incremental_F1.png")

print("\nFinished.")
