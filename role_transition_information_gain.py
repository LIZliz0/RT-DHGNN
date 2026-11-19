import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from scipy.stats import wilcoxon

import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("Loading...")
print("="*60)

df = pd.read_csv(
    "data/BPIC2017_org_enhanced.csv"
)

# 修复时间格式
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="mixed",
    utc=True
)

df = df.sort_values(
    ["case_id", "timestamp"]
).reset_index(drop=True)

# Org -> Role
df["role"] = df["org_label"]

print("Events :", len(df))
print("Cases  :", df["case_id"].nunique())
print("Roles  :", df["role"].nunique())

# Build Transition Samples

print("\nBuilding transitions...")

samples = []

for _, trace in df.groupby("case_id"):

    trace = trace.sort_values("timestamp")

    acts = trace["activity"].tolist()
    roles = trace["role"].tolist()

    for i in range(len(trace)-1):

        cur_act = acts[i]
        next_act = acts[i+1]

        cur_role = roles[i]
        next_role = roles[i+1]

        # 过滤自循环
        if cur_role == next_role:
            continue

        role_transition = (
            cur_role +
            "->" +
            next_role
        )

        samples.append(
            [
                cur_act,
                role_transition,
                next_act
            ]
        )

transition_df = pd.DataFrame(
    samples,
    columns=[
        "activity",
        "role_transition",
        "next_activity"
    ]
)

print("Transition Samples:",
      len(transition_df))

# Top20
top20 = (
    transition_df["role_transition"]
    .value_counts()
    .head(20)
)

print("\nTop20:")
print(top20)

# Mutual Information
print("\nComputing MI...")

mi_activity = mutual_info_score(
    transition_df["activity"],
    transition_df["next_activity"]
)

mi_role = mutual_info_score(
    transition_df["role_transition"],
    transition_df["next_activity"]
)

print(
    f"MI(Activity→Next)={mi_activity:.4f}"
)

print(
    f"MI(RoleTransition→Next)={mi_role:.4f}"
)

# Conditional Entropy
# H(Y|X)
def conditional_entropy(y, x):

    data = pd.DataFrame({
        "x": x,
        "y": y
    })

    total = len(data)

    ce = 0

    for xv, group in data.groupby("x"):

        p_x = len(group)/total

        pyx = (
            group["y"]
            .value_counts(normalize=True)
        )

        h = -np.sum(
            pyx*np.log2(pyx)
        )

        ce += p_x*h

    return ce

ce_activity = conditional_entropy(
    transition_df["next_activity"],
    transition_df["activity"]
)

ce_role = conditional_entropy(
    transition_df["next_activity"],
    transition_df["role_transition"]
)

print(
    f"H(Next|Activity)={ce_activity:.4f}"
)

print(
    f"H(Next|RoleTransition)={ce_role:.4f}"
)

# Random Forest
print("\nRandom Forest Validation")

X1 = transition_df[["activity"]]
X2 = transition_df[["role_transition"]]
X3 = transition_df[
    ["activity",
     "role_transition"]
]

y = transition_df["next_activity"]

# 编码
enc_act = LabelEncoder()
enc_role = LabelEncoder()
enc_y = LabelEncoder()

activity_code = enc_act.fit_transform(
    transition_df["activity"]
)

role_code = enc_role.fit_transform(
    transition_df["role_transition"]
)

y_code = enc_y.fit_transform(y)

X1 = activity_code.reshape(-1,1)

X2 = role_code.reshape(-1,1)

X3 = np.column_stack([
    activity_code,
    role_code
])

# RepeatedStratifiedKFold
rskf = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=5,
    random_state=42
)

acc1=[]
acc2=[]
acc3=[]

f11=[]
f12=[]
f13=[]

for train_idx,test_idx in rskf.split(
        X1,
        y_code
):

    y_train = y_code[train_idx]
    y_test = y_code[test_idx]

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    # activity

    rf.fit(
        X1[train_idx],
        y_train
    )

    pred = rf.predict(
        X1[test_idx]
    )

    acc1.append(
        accuracy_score(
            y_test,
            pred
        )
    )

    f11.append(
        f1_score(
            y_test,
            pred,
            average="macro"
        )
    )

    # role

    rf.fit(
        X2[train_idx],
        y_train
    )

    pred = rf.predict(
        X2[test_idx]
    )

    acc2.append(
        accuracy_score(
            y_test,
            pred
        )
    )

    f12.append(
        f1_score(
            y_test,
            pred,
            average="macro"
        )
    )

    # activity+role

    rf.fit(
        X3[train_idx],
        y_train
    )

    pred = rf.predict(
        X3[test_idx]
    )

    acc3.append(
        accuracy_score(
            y_test,
            pred
        )
    )

    f13.append(
        f1_score(
            y_test,
            pred,
            average="macro"
        )
    )

print("\nAccuracy")

print(np.mean(acc1))
print(np.mean(acc2))
print(np.mean(acc3))

print("\nMacro F1")

print(np.mean(f11))
print(np.mean(f12))
print(np.mean(f13))

# Wilcoxon

print("\nWilcoxon Test")

stat,p = wilcoxon(
    acc1,
    acc3
)

print("p =", p)

# =====================================================
# Figure1
# =====================================================

plt.figure(figsize=(12,8))

sns.barplot(
    x=top20.values,
    y=top20.index
)

plt.title(
    "Top20 Role Transition"
)

plt.tight_layout()

plt.savefig(
    "result/Fig1_Top20_RoleTransition.png",
    dpi=300
)

# Figure2
top_transition = top20.index[0]

temp = transition_df[
    transition_df["role_transition"]
    == top_transition
]

prob = (
    temp["next_activity"]
    .value_counts(normalize=True)
    .head(10)
)

plt.figure(figsize=(10,6))

sns.barplot(
    x=prob.values,
    y=prob.index
)

plt.title(
    f"Next Activity Distribution\n{top_transition}"
)

plt.tight_layout()

plt.savefig(
    "result/Fig2_Transition_NextActivity.png",
    dpi=300
)

# Figure3
plt.figure(figsize=(6,5))

plt.bar(
    ["Activity",
     "RoleTransition"],
    [mi_activity,
     mi_role]
)

plt.ylabel("Mutual Information")

plt.title(
    "Information Gain Comparison"
)

plt.tight_layout()

plt.savefig(
    "result/Fig3_InformationGain.png",
    dpi=300
)

# Figure4
perf = pd.DataFrame({

    "Feature":[
        "Activity",
        "RoleTransition",
        "Activity+Role"
    ],

    "Accuracy":[
        np.mean(acc1),
        np.mean(acc2),
        np.mean(acc3)
    ]
})

plt.figure(figsize=(8,5))

sns.barplot(
    data=perf,
    x="Feature",
    y="Accuracy"
)

plt.title(
    "Prediction Performance"
)

plt.tight_layout()

plt.savefig(
    "result/Fig4_RF_Performance.png",
    dpi=300
)

# Save Table
result = pd.DataFrame({

    "Metric":[
        "MI(Activity)",
        "MI(RoleTransition)",
        "H(Next|Activity)",
        "H(Next|RoleTransition)",
        "ACC(Activity)",
        "ACC(RoleTransition)",
        "ACC(Activity+Role)",
        "Wilcoxon_p"
    ],

    "Value":[
        mi_activity,
        mi_role,
        ce_activity,
        ce_role,
        np.mean(acc1),
        np.mean(acc2),
        np.mean(acc3),
        p
    ]
})

result.to_csv(
    "data/Table_InformationGain.csv",
    index=False
)

print("\nSaved:")

print("Fig1_Top20_RoleTransition.png")
print("Fig2_Transition_NextActivity.png")
print("Fig3_InformationGain.png")
print("Fig4_RF_Performance.png")
print("Table_InformationGain.csv")

print("\nFinished.")
