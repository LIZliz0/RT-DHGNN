import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 读取预处理后的数据
df = pd.read_csv("data/BPIC17_preprocessed.csv")

# 构建 Resource-Activity Matrix
resource_activity = pd.crosstab(
    df["resource"],
    df["activity"]
)

print("Resource-Activity Matrix Shape:")
print(resource_activity.shape)

# 标准化
scaler = StandardScaler()

X_scaled = scaler.fit_transform(resource_activity)

# PCA降维（推荐）
# 保留95%信息
pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA Shape:")
print(X_pca.shape)

# 测试不同K值
k_range = range(2, 16)

silhouette_scores = []
inertias = []

print("\n===== K Selection Results =====")

for k in k_range:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_pca)

    # Silhouette Score
    score = silhouette_score(X_pca, labels)

    # SSE
    inertia = kmeans.inertia_

    silhouette_scores.append(score)
    inertias.append(inertia)

    print(f"K = {k:<2} | Silhouette Score = {score:.4f}")

# 选择最佳K
best_k = k_range[
    silhouette_scores.index(max(silhouette_scores))
]

print("\n================================")
print(f"Optimal K = {best_k}")
print("================================")

# 可视化 Silhouette Score
plt.figure(figsize=(10, 5))

plt.plot(
    list(k_range),
    silhouette_scores,
    marker='o'
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Optimal K Selection using Silhouette Score")

plt.grid(True)

plt.savefig(
    "result/silhouette_score.png",
    dpi=300,
    bbox_inches='tight'
)

print("Silhouette figure saved.")

# 使用最佳K重新聚类
final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

final_labels = final_kmeans.fit_predict(X_pca)

# 构建 Role Label
resource_activity["role_cluster"] = final_labels

resource_activity["role_label"] = (
    resource_activity["role_cluster"]
    .apply(lambda x: f"Role{x+1}")
)

# 映射回原始日志
role_map = resource_activity["role_label"].to_dict()

df["role_label"] = df["resource"].map(role_map)

# 保存最终结果
df.to_csv(
    "data/BPIC2017_role_enhanced.csv",
    index=False
)

print("\nSaved:")
print("data/BPIC2017_role_enhanced.csv")
