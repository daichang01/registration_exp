import numpy as np
import open3d as o3d
import copy
from scipy.spatial import KDTree
from scipy.linalg import logm, expm
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from pyquaternion import Quaternion
from joblib import Parallel, delayed
from numba import njit

from src.utils import *




def compute_pca(points):
    pca = PCA(n_components=3)
    pca.fit(points - np.mean(points, axis=0))
    eigenvectors = pca.components_.T  # 列向量为特征向量
    
    # 确保特征向量构成右手系（叉积第三条=第三条）
    if not np.allclose(np.cross(eigenvectors[:,0], eigenvectors[:,1]), eigenvectors[:,2], atol=1e-6):
        eigenvectors[:,2] = np.cross(eigenvectors[:,0], eigenvectors[:,1])  # 强制纠正
    return eigenvectors, np.mean(points, axis=0)

def compute_pca_fast(points):
    mean = np.mean(points, axis=0)
    centered = points - mean
    
    # 向量化计算协方差（比循环更快）
    cov = (centered.T @ centered) / len(points)  # (3, N) @ (N, 3) → (3,3)
    
    # 计算特征向量（eigh 比 eig 快，因为协方差矩阵对称）
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # 按特征值降序排列
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    
    # 强制右手系
    if not np.allclose(np.cross(eigenvectors[:,0], eigenvectors[:,1]), eigenvectors[:,2], atol=1e-6):
        eigenvectors[:,2] = np.cross(eigenvectors[:,0], eigenvectors[:,1])
    
    return eigenvectors, mean





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



@njit
def slerp_core(q1, q2, t):
    dot = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    
    w1 = np.sin((1-t)*theta) / sin_theta
    w2 = np.sin(t*theta) / sin_theta
    return w1*q1 + w2*q2

def approximate_slerp(q1, q2):
    """
    用泰勒展开近似 SLERP（速度比 SLERP 快 2 倍，精度更高）
    适用于夹角 < 90°（即 dot(q1, q2) > 0）
    """
    dot = np.dot(q1.q, q2.q)
    if dot < 0:  # 处理符号一致性
        q2_q = -q2.q
        dot = -dot
    else:
        q2_q = q2.q
    
    # 小角度时用泰勒展开近似 sin(theta)/theta ≈ 1 - theta²/6
    if dot > 0.95:  # 夹角 < 18°
        theta = np.sqrt(1 - dot * dot)  # sin(theta)
        w2 = 0.5 - theta * theta / 12  # 泰勒展开近似
        w1 = 1 - w2
    else:  # 大角度时回退到线性平均
        w1 = w2 = 0.5
    
    q_avg = w1 * q1.q + w2 * q2_q
    q_avg /= np.linalg.norm(q_avg)
    
    return Quaternion(q_avg)


def calculate_rmse_with_matching(points_a, points_b):
    # 为points_b构建KD树
    tree = KDTree(points_b)
    # 查询每个points_a点到points_b的最近邻距离
    distances, _ = tree.query(points_a)
    return np.sqrt(np.mean(distances ** 2))  # RMSE

def pca_double_adjust_parallel(source_cloud, target_cloud, source_pca, target_pca,source_pca2, target_pca2,n_jobs=-1):
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

    def process_sign(signs):
        print(f"\n尝试符号组合: {signs}")
        # 根据符号组合调整源点云的特征向量方向
        adjusted_target_left = target_left_eigenvectors * signs
        adjusted_target_right = target_right_eigenvectors * signs

        R_left = adjusted_target_left @ source_left_eigenvectors.T
        R_right = adjusted_target_right @ source_right_eigenvectors.T

        if not is_matrix_sane(R_left) or not is_matrix_sane(R_right):
            print("矩阵数值异常，跳过")
            return (float('inf'), None, None)

        R_left = ensure_rotation_matrix(R_left)
        R_right = ensure_rotation_matrix(R_right)


        if not is_valid_rotation_matrix(R_left) or not is_valid_rotation_matrix(R_right):
            print(f"跳过无效旋转矩阵的符号组合: {signs}")
            return (float('inf'), None, None)

        t_left = target_left_centroid - R_left @ source_left_centroid
        t_right = target_right_centroid - R_right @ source_right_centroid

        
        # ---- 平均变换（基于四元数球面平均）----------------
        # 将旋转矩阵转换为四元数
        try:
            q_left = Quaternion(matrix=R_left)
            q_right = Quaternion(matrix=R_right)
        except Exception as e:
            print(f"Skipping signs {signs} due to invalid rotation: {e}")
            return (float('inf'), None, None)
        q_avg = average_quaternions(q_left, q_right)

        
        
        R_avg = q_avg.rotation_matrix


        # 平均位移
        t_avg = (t_left + t_right) * 0.5

        # ---- 评估当前符号组合的配准质量 ---------------------
        transformed_points = (R_avg @ source_points.T).T + t_avg
        # current_rmse = np.sqrt(np.mean(np.sum((transformed_points - target_points)**2, axis=1)))
        current_rmse = calculate_rmse_with_matching(transformed_points, target_points)
        return (current_rmse, R_avg, t_avg)
    
        # 并行处理所有符号组合
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_sign)(signs) for signs in sign_combinations
    )

    # 找出RMSE最小的结果
    min_rmse, best_R, best_t = min(results, key=lambda x: x[0])
    
    
    coarse_transformation = np.eye(4)
    coarse_transformation[:3, :3] = best_R
    coarse_transformation[:3, 3] = best_t

    return coarse_transformation

def fast_quaternion_average(q1, q2):
    """快速线性平均四元数（精度损失约0.1-0.5%，速度快2倍）"""
    q_avg_vector = (q1.vector + q2.vector) / 2  # 直接线性平均
    q_avg = Quaternion(vector=q_avg_vector).normalised  # 归一化
    return q_avg

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




def bar_tw_icp_registration(source, target, init_transform=np.eye(4), 
                           max_iter=50, tau=1.0, eta=0.95, c=4.685, gamma=0.5,
                           convergence_threshold=1e-6):
    """
    双向自适应鲁棒加权ICP配准算法
    
    参数:
        source (o3d.geometry.PointCloud): 源点云（如术前规划点云）
        target (o3d.geometry.PointCloud): 目标点云（如术中采集点云）
        init_transform (np.array): 粗配准得到的初始变换矩阵 (4x4)
        max_iter (int): 最大迭代次数
        tau (float): 双向搜索距离阈值
        eta (float): 阈值衰减系数(0.9~0.99)
        c (float): Tukey权重调节因子
        gamma (float): Hausdorff距离缩放因子
        convergence_threshold (float): 收敛条件
        
    返回:
        o3d.registration.RegistrationResult: 包含最终变换矩阵和评估指标
    """
    source_temp = copy.deepcopy(source)
    
    # 初始变换
    source_temp = source.transform(init_transform)
    source_points = np.asarray(source_temp.points)
    target_points = np.asarray(target.points)
    N, M = len(source_points), len(target_points)

    # 初始化鲁棒参数
    sigma = gamma * compute_hausdorff(source_points, target_points)
    prev_rmse = np.inf
    
    # 构建目标点云的KDTree (反向验证用)
    target_kdtree = cKDTree(target_points)
    source_kdtree = cKDTree(source_points)

    current_transform = init_transform.copy()
    current_rmse = float('inf')  # 添加变量初始化
    history = []

    for iteration in range(max_iter):
        ### 阶段1: 双向几何一致性验证 ###
        # 正向搜索：source -> target
        _, fw_indices = target_kdtree.query(source_points, k=1, eps=0)
        fw_pairs = list(zip(range(N), fw_indices))
        
        # 反向验证：target点反向查找source
        valid_pairs = []
        for s_idx, t_idx in fw_pairs:
            query_point = target_points[t_idx]
            # 反向搜索最近邻
            bv_dist, bv_idx = source_kdtree.query(query_point, k=1, eps=0)
            if bv_idx == s_idx and bv_dist < tau:
                valid_pairs.append( (s_idx, t_idx) )
                # print(f"s_idx, t_idx:{s_idx, t_idx}")
        
        if len(valid_pairs) == 0:
            print(f"Iter {iteration}: 没有有效匹配点对!")
            break
        print(f"len(valid_pairs):{len(valid_pairs)}")
        ### 阶段2: 计算残差与权重 ###
        src_corr = source_points[[p[0] for p in valid_pairs]]
        tgt_corr = target_points[[p[1] for p in valid_pairs]]
        
        # 应用当前变换
        transformed_src = np.dot(src_corr, current_transform[:3, :3].T) + current_transform[:3, 3]
        residuals = np.linalg.norm(transformed_src - tgt_corr, axis=1)
        print(f"residuals:{residuals}")
        
        # 计算鲁棒权重：Tukey双权函数
        residual_scale = np.median(residuals)  # 更鲁棒的尺度估计
        sigma_current = max(sigma * (eta**iteration), 1e-6)  # 防止除以零
        print(f"sigma_current:{sigma_current}")
        
        # Tukey权重计算
        adjusted_res = residuals / (c * sigma_current)
        weights = np.where(adjusted_res <= 1.0, 
                          (1 - adjusted_res**2)**2, 
                          0.0)  # Eq.(6)
        print(f"weights:{weights}")
        total_weight = np.sum(weights)
        print(f"total_weight:{total_weight}")
        if total_weight < 1e-7: 
            current_rmse = float('inf')  # 确保变量赋值
            print(f"Iter {iteration}: 权重极低，终止迭代")
            break
        
        ### 阶段3: 加权SVD求解刚体变换 ###
        src_centroid = np.average(src_corr, axis=0, weights=weights)
        tgt_centroid = np.average(tgt_corr, axis=0, weights=weights)
        
        src_centered = src_corr - src_centroid
        tgt_centered = tgt_corr - tgt_centroid
        
        W = np.diag(weights)
        S = src_centered.T @ W @ tgt_centered  # 加权协方差矩阵
        
        # SVD分解求解旋转
        U, _, VT = np.linalg.svd(S)
        R = VT.T @ U.T
        if np.linalg.det(R) < 0:  # 处理反射情况
            VT[-1, :] *= -1
            R = VT.T @ U.T
        
        # 平移计算
        t = tgt_centroid - R @ src_centroid
        
        # 更新全局变换
        delta_transform = np.eye(4)
        delta_transform[:3, :3] = R
        delta_transform[:3, 3] = t
        current_transform = delta_transform @ current_transform
        
        # 收敛判断
        delta_rot = np.linalg.norm(R - np.eye(3), 'fro')
        delta_trans = np.linalg.norm(t)
        if delta_rot < convergence_threshold and delta_trans < convergence_threshold:
            print(f"Iter {iteration}: 收敛!")
            break
        
        ### 收据收集与显示 ###
        current_rmse = np.sqrt(np.sum(weights * residuals**2) / total_weight)
        history.append({'iter': iteration, 'rmse': current_rmse, 
                       'num_valid': len(valid_pairs), 'sigma': sigma_current})
        prev_rmse = current_rmse
    
    ### 返回Open3D标准格式 ###
    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = current_transform
    result.inlier_rmse = current_rmse
    result.fitness = len(valid_pairs) / min(N, M)  # 内点比例
    
    return result, history

def compute_hausdorff(source, target):
    """计算Hausdorff距离的近似值（Open3D优化版本）"""
    dist1 = np.max( o3d.geometry.PointCloud.compute_point_cloud_distance(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source)),
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target)) ))
    dist2 = np.max( o3d.geometry.PointCloud.compute_point_cloud_distance(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target)),
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source)) ))
    return max(dist1, dist2)

def compute_hausdorff_fine(source_points, target_points):
    """计算对称Hausdorff距离"""
    src_tree = cKDTree(source_points)
    tgt_tree = cKDTree(target_points)
    dist1, _ = src_tree.query(target_points, k=1)  # target到source的距离
    dist2, _ = tgt_tree.query(source_points, k=1)  # source到target的距离
    return max(np.max(dist1), np.max(dist2))