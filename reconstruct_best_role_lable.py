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

# 使用K=8聚类
final_kmeans = KMeans(
    n_clusters=8,
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
print("BPIC2017_role_enhanced.csv")

