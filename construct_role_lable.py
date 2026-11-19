# 必要库
import pandas as pd
import pm4py
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# 读取 BPIC2017 XES 文件（支持 .xes.gz）
log = pm4py.read_xes("data/BPI Challenge 2017.xes.gz")

print(f"Total cases loaded: {len(log)}")

#提取事件字段
#如果 log 是 DataFrame，直接使用
if isinstance(log, pd.DataFrame):
    df = log.copy()
    print("数据已加载为 DataFrame")
    print(f"列名: {df.columns.tolist()}")

    #重命名列以匹配我们的需求
    #确保有 case_id 列
    if 'case:concept:name' in df.columns:
        df.rename(columns={'case:concept:name': 'case_id'}, inplace=True)
    elif 'case_id' not in df.columns:
        # 如果没有 case_id 列，使用索引
        df['case_id'] = df.index

    #确保活动列存在
    if 'concept:name' in df.columns:
        df.rename(columns={'concept:name': 'activity'}, inplace=True)
    elif 'activity' not in df.columns:
        df['activity'] = 'Unknown'

    #确保时间戳列存在
    if 'time:timestamp' in df.columns:
        df.rename(columns={'time:timestamp': 'timestamp'}, inplace=True)
    elif 'timestamp' not in df.columns:
        df['timestamp'] = pd.NaT

    #确保资源列存在
    if 'org:resource' in df.columns:
        df.rename(columns={'org:resource': 'resource'}, inplace=True)
    elif 'resource' not in df.columns:
        df['resource'] = 'Unknown'

    #确保生命周期列存在
    if 'lifecycle:transition' in df.columns:
        df.rename(columns={'lifecycle:transition': 'lifecycle'}, inplace=True)
    elif 'lifecycle' not in df.columns:
        df['lifecycle'] = 'Unknown'

else:
    #如果 log 是传统的 EventLog 对象
    data = []
    for case in log:
        case_id = case.attributes.get('concept:name', f"Case_{len(data)}")
        for event in case:
            activity = event.get("concept:name", "Unknown")
            resource = event.get("org:resource", "Unknown")
            timestamp = event.get("time:timestamp", pd.NaT)
            lifecycle = event.get("lifecycle:transition", "Unknown")

            data.append({
                "case_id": case_id,
                "activity": activity,
                "resource": resource,
                "timestamp": timestamp,
                "lifecycle": lifecycle
            })
    df = pd.DataFrame(data)

print(f"\nDataFrame shape: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"Unique cases: {df['case_id'].nunique()}")
print(f"Unique activities: {df['activity'].nunique()}")
print(f"Unique resources: {df['resource'].nunique()}")

#处理缺失资源
df["resource"] = df["resource"].fillna("Unknown")

#构造角色标签
le = LabelEncoder()
df["resource_id"] = le.fit_transform(df["resource"])

#根据资源数量动态调整聚类数
n_resources = df['resource_id'].nunique()
n_clusters = min(10, n_resources)  #暂时假设10个，但不能超过资源数

print(f"\n实际资源数: {n_resources}")
print(f"聚类数: {n_clusters}")

if n_clusters > 1:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["role_label"] = kmeans.fit_predict(df[["resource_id"]])
    df["role_label"] = df["role_label"].apply(lambda x: f"Role{x + 1}")
else:
    # <UNK_ROLE>作为职责流层的特殊兜底节点，仅用于承接无法映射至已有角色的执行资源
    df["role_label"] = "<UNK_ROLE>"

#按 case_id + timestamp 排序
if 'timestamp' in df.columns and not df['timestamp'].isna().all():
    df = df.sort_values(by=["case_id", "timestamp"]).reset_index(drop=True)
else:
    df = df.sort_values(by=["case_id"]).reset_index(drop=True)

#输出检查
print("\n=== 数据预览 ===")
print(df.head())

print("\n=== 每个Case的事件数统计 ===")
case_stats = df.groupby("case_id").size()
print(case_stats.describe())

print("\n=== 活动分布 ===")
print(df['activity'].value_counts().head(10))

print("\n=== 资源分布 ===")
print(df['resource'].value_counts().head(10))

print("\n=== 生命周期状态分布 ===")
if 'lifecycle' in df.columns:
    print(df['lifecycle'].value_counts())
else:
    print("无生命周期信息")

#保存预处理结果
df.to_csv("data/BPIC17_preprocessed.csv", index=False)
print("\n✅ 预处理数据已保存到 BPIC17_preprocessed.csv")

#统计活动在不同角色的分布
if 'role_label' in df.columns:
    activity_role_count = df.groupby(["activity", "role_label"]).size().reset_index(name="count")
    print("\n=== 活动在各角色的分布（前20）===")
    print(activity_role_count.sort_values("count", ascending=False).head(20))

    print("\n=== 各角色活动数量统计 ===")
    role_stats = df.groupby("role_label").agg({
        'activity': 'count',
        'case_id': 'nunique'
    }).rename(columns={'activity': 'total_events', 'case_id': 'unique_cases'})
    print(role_stats)
