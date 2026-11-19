import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取 CSV
df = pd.read_csv("data/BPIC2017_role_enhanced.csv")

# 构建角色-活动热力图矩阵
heatmap_data = pd.crosstab(
    df["role_label"],
    df["activity"]
)

# 统计 role 下 activity 分布
role_activity = (
    df.groupby(["role_label", "activity"])
    .size()
    .reset_index(name="count")
)

# 查看每个角色最主要活动
for role in sorted(df["role_label"].unique()):

    print("\n====================")
    print(role)

    top_activities = (
        role_activity[
            role_activity["role_label"] == role
        ]
        .sort_values("count", ascending=False)
        .head(10)
    )

    print(top_activities)

# 画热力图
plt.figure(figsize=(14,6))

sns.heatmap(
    heatmap_data,
    cmap="YlGnBu"
)

plt.title("Role-Activity Heatmap")
plt.xlabel("Activity")
plt.ylabel("Role")

plt.tight_layout()
plt.savefig(
    "role_activity_heatmap.png",
    dpi=300
)

print("Heatmap saved.")

# 每簇资源数（表 X "资源数"列）
print(df.groupby('role_label')['resource'].nunique())

# 训练集口径的角色-活动统计
cases = sorted(df['case_id'].unique()); n = len(cases)
train_cases = set(cases[:int(n*0.7)])
tr = df[df['case_id'].isin(train_cases)]
ct = pd.crosstab(tr['role_label'], tr['activity'])
for r in ct.index:
    top = ct.loc[r].sort_values(ascending=False).head(5)
    share = (ct.loc[r] / ct.loc[r].sum())
    print(r, [(a, int(c), f"{share[a]:.1%}") for a, c in top.items()])