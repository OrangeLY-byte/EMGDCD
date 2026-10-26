import numpy as np
import torch
from torchmetrics.classification import AUROC
import torch.nn.functional as F

def compute_matrix_auroc(pred_matrix: torch.Tensor, label_matrix: torch.Tensor) -> float:
    """
    计算给定 N×N 概率矩阵和标签矩阵的 AUROC。

    Args:
        pred_matrix (torch.Tensor): 模型预测概率矩阵，元素应在 [0,1]。
        label_matrix (torch.Tensor): 标签矩阵，只含 0 或 1。

    Returns:
        float: AUROC 值（标量）
    """
    # 展平成一维向量
    preds = pred_matrix.flatten()
    targets = label_matrix.flatten()
    # 初始化 AUROC 指标器
    auroc_metric = AUROC(task="binary")
    # 计算 AUROC
    auroc = auroc_metric(preds, targets)
    return auroc.item()

def shd_distance(A: np.ndarray, B: np.ndarray):
    """
    A: 概率图 (N×N)，float，取值在 [0,1]
    B: 二值图 (N×N)，int 或 bool，取值为 0 或 1
    threshold: 概率图的二值化阈值
    返回：
    - 结构汉明距离（SHD）：int
    """
    A = A.astype(int)
    B = B.astype(int)
    shd = np.sum(A != B)
    return int(shd)


def cosine_similarity(A: torch.Tensor, B: torch.Tensor,n_nodes) -> float:
    """
    计算两个矩阵展平后的一维余弦相似度。

    参数：
        A (Tensor): 任意形状的张量
        B (Tensor): 与 A 形状相同的张量

    返回：
        float: A 与 B 展平后的一维余弦相似度（标量）
    """
    A_flat = A.float().reshape(-1)
    B_flat = B.float().reshape(n_nodes*n_nodes)
    sim = F.cosine_similarity(A_flat.unsqueeze(0), B_flat.unsqueeze(0), dim=1)
    return sim.item()

