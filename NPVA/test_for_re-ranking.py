import torch
import torchreid
from torchreid.reid.utils.rerank import re_ranking



# 假你已经有了特征向量
# qf：查询集的特征向量
# gf：库集的特征向量

# 示例：使用随机生成的特征向量
qf = torch.rand(10, 512)  # 10个查询样本，每个样本512维特征
gf = torch.rand(100, 512)  # 100个库样本，每个样本512维特征

# 计算距离矩阵（欧氏距离或者余弦距离）
# distmat: 查询集和库集之间的距离矩阵
distmat = torch.cdist(qf, gf, p=2)  # 欧氏距离

# 执行K-reciprocal re-ranking
k1, k2, lambda_value = 20, 6, 0.3
reranked_distmat = utils.rerank.re_ranking(distmat, k1=k1, k2=k2, lambda_value=lambda_value)

print("Original Distance Matrix:")
print(distmat)
print("\nRe-ranked Distance Matrix:")
print(reranked_distmat)
