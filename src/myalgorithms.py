import numpy as np
import open3d as o3d
import copy
from scipy.spatial import KDTree
from scipy.linalg import logm, expm
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from pyquaternion import Quaternion



# def compute_pca(points):
#         # 计算点集的中心点。这里使用np.mean计算所有点的平均值，axis=0确保按列求平均（即对每个维度求平均）。
#         centroid = np.mean(points, axis=0)
        
#         # 中心化点云：将每个点的坐标减去中心点的坐标，使得新的点云集中在原点附近。
#         centered_points = points - centroid
        
#         # 计算中心化后点云的协方差矩阵。np.cov用于计算协方差矩阵，参数.T表示转置，因为np.cov默认是按行处理的。
#         cov_matrix = np.cov(centered_points.T)
        
#         # 使用np.linalg.eigh计算协方差矩阵的特征值和特征向量。eigh是专为对称或厄米特矩阵设计的，更稳定。
#         eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
#         # 对特征值进行降序排序，获取排序后的索引。np.argsort对特征值数组进行排序，默认升序，[::-1]实现降序。
#         idx = np.argsort(eigenvalues)[::-1]
        
#         # 重排特征向量，使其与特征值的降序对应。这确保了第一个特征向量对应最大的特征值。
#         eigenvectors = eigenvectors[:, idx]

#         # 确保特征向量的方向一致性，例如，保持右手法则
#         if np.linalg.det(eigenvectors) < 0:
#             eigenvectors[:, 2] = -eigenvectors[:, 2]

#             # 打印出最大的三个特征值，和相应的特征向量
#         # print("length of eigenvectors:", eigenvectors.shape[1])
#         # print("Top 3 eigenvalues:", eigenvalues[:3])
#         # print("Corresponding eigenvectors:\n", eigenvectors[:, :3])
        
#         # 返回排序后的特征向量和中心点。特征向量的每一列都是一个主成分方向。
#         return eigenvectors, centroid

# 在 compute_pca 函数内添加验证
def compute_pca(points):
    pca = PCA(n_components=3)
    pca.fit(points - np.mean(points, axis=0))
    eigenvectors = pca.components_.T  # 列向量为特征向量
    
    # 确保特征向量构成右手系（叉积第三条=第三条）
    if not np.allclose(np.cross(eigenvectors[:,0], eigenvectors[:,1]), eigenvectors[:,2], atol=1e-6):
        eigenvectors[:,2] = np.cross(eigenvectors[:,0], eigenvectors[:,1])  # 强制纠正
    return eigenvectors, np.mean(points, axis=0)



def compute_transform(source_pcd, target_pcd):
    """计算源点云到目标点云的变换矩阵"""
    # 计算 PCA 主方向
    source_eigenvectors, source_centroid = compute_pca(source_pcd)
    target_eigenvectors, target_centroid = compute_pca(target_pcd)
    # 计算旋转矩阵
    R = target_eigenvectors @ source_eigenvectors.T
    # 计算平移向量
    t = target_centroid - R @ source_centroid
    return R, t

def transform_points(points, R, t):
    transformed_points = np.dot(points, R.T) + t
    return transformed_points


def calculate_rmse(source_points, target_points):
        # 使用 float64 确保高精度计算
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    # 创建目标点云的KD树
    tree = cKDTree(target_points)
    # 查询源点云中每个点在目标点云中的最近邻点
    distances, indices = tree.query(source_points, k=1)
    # 找到每个源点云点对应的最近的目标点云点
    nearest_target_points = target_points[indices]
    # 计算源点云和最近的目标点云点之间的均方误差 (MSE)
    mse = np.mean((source_points - nearest_target_points)**2)
    rmse = np.sqrt(mse)
    return rmse * 1000


def ensure_rotation_matrix(R, eps=1e-8):
    """增强版旋转矩阵修正，防止数值不稳定"""
    # 先检查输入是否接近正交
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        R = np.eye(3)  # 极端情况下重置为单位矩阵
    
    # SVD分解
    try:
        U, s, Vt = np.linalg.svd(R)
        R_corrected = U @ Vt
        if np.linalg.det(R_corrected) < 0:
            U[:, -1] *= -1
            R_corrected = U @ Vt
        return R_corrected
    except np.linalg.LinAlgError:
        return np.eye(3)  # 完全失败时返回单位矩阵



def is_valid_rotation_matrix(R, tol=1e-6):
    """检查矩阵是否有效（正交且 det(R)=1）"""
    return (
        np.allclose(R.T @ R, np.eye(3), atol=tol) and 
        np.isclose(np.linalg.det(R), 1.0, atol=tol)
    )

def is_matrix_sane(R):
    """检查矩阵元素是否在合理范围"""
    return np.all(np.abs(R) < 10) and not np.any(np.isnan(R))

def average_quaternions(q1, q2):
    """手动计算两个四元数的平均（球面线性插值）"""
    # 确保四元数的实部（w）符号一致
    if np.dot(q1.q, q2.q) < 0:
        q2 = -q2  # 反转其中一个四元数
    q_avg = q1.q + q2.q  # 简单求和
    q_avg = q_avg / np.linalg.norm(q_avg)  # 归一化
    return Quaternion(q_avg)

def calculate_rmse_with_matching(points_a, points_b):
    # 为points_b构建KD树
    tree = KDTree(points_b)
    # 查询每个points_a点到points_b的最近邻距离
    distances, _ = tree.query(points_a)
    return np.sqrt(np.mean(distances ** 2))  # RMSE

def pca_double_adjust(source_cloud, target_cloud, source_pca, target_pca,source_pca2, target_pca2):
    if target_pca is None or target_pca2 is None:
        print("target_pca is None")
        return None
    # 将源点云和目标点云的点转换为NumPy数组
    source_points = np.asarray(source_cloud.points)
    target_points = np.asarray(target_cloud.points)
    source_left_points = np.asarray(source_pca.points)
    target_left_points = np.asarray(target_pca.points)
    source_right_points = np.asarray(source_pca2.points)
    target_right_points = np.asarray(target_pca2.points)

 

    source_left_eigenvectors, source_left_centroid = compute_pca(source_left_points)
    target_left_eigenvectors, target_left_centroid = compute_pca(target_left_points)
    source_right_eigenvectors, source_right_centroid = compute_pca(source_right_points)
    target_right_eigenvectors, target_right_centroid = compute_pca(target_right_points)
    

    # 生成所有可能的符号组合 (2^3=8种可能性)
    sign_combinations = [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1),
                       (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)]
    
    min_rmse = float('inf')
    best_R, best_t = None, None

   
    for signs in sign_combinations:
        print(f"\n尝试符号组合: {signs}")
        # 根据符号组合调整源点云的特征向量方向
        adjusted_target_left = target_left_eigenvectors * signs
        adjusted_target_right = target_right_eigenvectors * signs

        R_left = adjusted_target_left @ source_left_eigenvectors.T
        R_right = adjusted_target_right @ source_right_eigenvectors.T

        if not is_matrix_sane(R_left) or not is_matrix_sane(R_right):
            print("矩阵数值异常，跳过")
            continue


        R_left = ensure_rotation_matrix(R_left)
        R_right = ensure_rotation_matrix(R_right)


        if not is_valid_rotation_matrix(R_left) or not is_valid_rotation_matrix(R_right):
            print(f"跳过无效旋转矩阵的符号组合: {signs}")
            continue  # 直接跳过当前组合

        t_left = target_left_centroid - R_left @ source_left_centroid
        t_right = target_right_centroid - R_right @ source_right_centroid

        
        # ---- 平均变换（基于四元数球面平均）----------------
        # 将旋转矩阵转换为四元数
        try:
            q_left = Quaternion(matrix=R_left)
            q_right = Quaternion(matrix=R_right)
        except Exception as e:
            print(f"Skipping signs {signs} due to invalid rotation: {e}")
            continue
        q_avg = average_quaternions(q_left, q_right)
        R_avg = q_avg.rotation_matrix

        # 平均位移
        t_avg = (t_left + t_right) * 0.5

        # ---- 评估当前符号组合的配准质量 ---------------------
        transformed_points = (R_avg @ source_points.T).T + t_avg
        # current_rmse = np.sqrt(np.mean(np.sum((transformed_points - target_points)**2, axis=1)))
        current_rmse = calculate_rmse_with_matching(transformed_points, target_points)

        if current_rmse < min_rmse:
            min_rmse = current_rmse
            best_R, best_t = R_avg, t_avg
    
    coarse_transformation = np.eye(4)
    coarse_transformation[:3, :3] = best_R
    coarse_transformation[:3, 3] = best_t

    return coarse_transformation





def pca_double_adjust_old(source, target,source_pca=0, target_pca=0,source_pca2=0, target_pca2=0):
    # 获取点云数据
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    # 计算源点云和目标点云的质心
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    # 将点云中心化（减去质心）
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    # 计算源点云和目标点云的协方差矩阵
    source_cov = np.cov(source_centered.T)
    target_cov = np.cov(target_centered.T)

    # 计算协方差矩阵的特征值和特征向量
    source_eigvals, source_eigvecs = np.linalg.eigh(source_cov)
    target_eigvals, target_eigvecs = np.linalg.eigh(target_cov)

    # 对特征向量进行排序（按特征值从大到小）
    source_eigvecs = source_eigvecs[:, np.argsort(-source_eigvals)]
    target_eigvecs = target_eigvecs[:, np.argsort(-target_eigvals)]

    # 确保特征向量方向正确（避免符号模糊性问题）
    for i in range(3):
        if np.dot(source_eigvecs[:, i], target_eigvecs[:, i]) < 0:
            target_eigvecs[:, i] *= -1

    # 计算旋转矩阵
    R = target_eigvecs @ source_eigvecs.T

    # 计算平移向量
    t = target_centroid - R @ source_centroid

    # 构造变换矩阵
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t

    return transformation