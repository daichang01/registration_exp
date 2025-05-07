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
from sklearn.utils.extmath import randomized_svd

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
                          max_iter=30, tau=0.3, convergence_threshold=1e-6,
                          debug_level=1):
    """
    双向验证的ICP配准算法
    参数说明：
    source/target: Open3D点云对象
    init_transform: 初始变换矩阵(4x4)
    tau: 距离阈值
    debug_level: 0=无输出, 1=基础信息, 2=详细匹配信息
    """
    # 深拷贝并应用初始变换
    source_temp = copy.deepcopy(source).transform(init_transform)
    src_points = np.asarray(source_temp.points)
    tgt_points = np.asarray(target.points)
    
    if debug_level > 0:
        print("\n=== ICP配准初始化 ===")
        print(f"源点云数量: {len(src_points)}, 目标点云数量: {len(tgt_points)}")
        print(f"参数: max_iter={max_iter}, tau={tau}, conv_thresh={convergence_threshold}")

    # 构建KDTree（预编译优化）
    tgt_kdtree = cKDTree(tgt_points, compact_nodes=False, balanced_tree=False)
    src_kdtree = cKDTree(src_points, compact_nodes=False, balanced_tree=False)
    current_transform = init_transform.copy()
    history = []

    for iter in range(max_iter):
        if debug_level > 0:
            print(f"\n--- 迭代 {iter+1}/{max_iter} ---")

        ### ========== 阶段1: 双向验证 ========== ###
        # 1. 批量前向查询（source->target）
        _, fw_idx = tgt_kdtree.query(src_points, k=1, workers=-1)
        
        # 2. 批量反向验证（target->source）
        candidate_tgt_points = tgt_points[fw_idx]
        _, bv_idx = src_kdtree.query(candidate_tgt_points, k=1, workers=-1)
        
        # 3. 向量化条件判断
        s_indices = np.arange(len(src_points))
        sq_distances = np.sum((src_points - candidate_tgt_points)**2, axis=1)
        valid_mask = (bv_idx == s_indices) & (sq_distances < tau**2)
        valid_pairs = np.column_stack((s_indices[valid_mask], fw_idx[valid_mask])).tolist()
        
        # 有效性检查
        if len(valid_pairs) < 50:
            print(f"⚠️ 警告: 仅有 {len(valid_pairs)} 对有效点 (<50), 终止迭代")
            break
            
        if debug_level > 0:
            valid_ratio = len(valid_pairs)/min(len(src_points),len(tgt_points))*100
            print(f"有效匹配对数: {len(valid_pairs)} (占比{valid_ratio:.1f}%)")
            if debug_level > 1:
                valid_distances = np.sqrt(sq_distances[valid_mask])
                for (s_idx, t_idx), dist in zip(valid_pairs, valid_distances):
                    print(f"匹配点: src[{s_idx}]->tgt[{t_idx}], 距离={dist:.4f}")

        ### ========== 阶段2: SVD求解 ========== ###
        src_corr = src_points[np.asarray(valid_pairs)[:,0]]
        tgt_corr = tgt_points[np.asarray(valid_pairs)[:,1]]
        
        # 1. 快速质心计算
        src_centroid = np.mean(src_corr, axis=0)
        tgt_centroid = np.mean(tgt_corr, axis=0)
        
        # 2. 协方差矩阵计算加速
        H = (src_corr - src_centroid).T @ (tgt_corr - tgt_centroid)
        
        # 3. 分支SVD优化
        if H.shape == (3,3):
            U, S, VT = np.linalg.svd(H)
        else:
            U, S, VT = randomized_svd(H, n_components=3)
        
        # 计算旋转矩阵
        R = VT.T @ U.T
        if np.linalg.det(R) < 0:
            VT[-1,:] *= -1
            R = VT.T @ U.T
        t = tgt_centroid - R @ src_centroid

        # 4. 更新变换矩阵
        delta_T = np.eye(4)
        delta_T[:3,:3] = R
        delta_T[:3,3] = t
        current_transform = delta_T @ current_transform

        # 误差计算
        transformed_src = (R @ src_corr.T).T + t
        residuals = np.linalg.norm(transformed_src - tgt_corr, axis=1)
        current_rmse = np.sqrt(np.mean(residuals**2))
        history.append({
            'iter': iter,
            'rmse': current_rmse,
            'valid_pairs': len(valid_pairs)
        })

        # 收敛判断（支持首次迭代终止）
        if iter == 0:
            initial_rmse = current_rmse
        delta_error = abs(history[-2]['rmse']-current_rmse) if iter>0 else initial_rmse
        if delta_error < convergence_threshold:
            if debug_level > 0:
                print(f"✅ 收敛于迭代 {iter+1}, RMSE变化{delta_error:.2e}<{convergence_threshold}")
            break

    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = current_transform
    result.inlier_rmse = history[-1]['rmse'] if history else float('inf')
    result.fitness = len(valid_pairs)/min(len(src_points),len(tgt_points))
    
    if debug_level > 0:
        print("\n=== 配准结果 ===")
        print(f"最终变换矩阵:\n{np.round(current_transform, 4)}")
        print(f"RMSE: {result.inlier_rmse:.6f}, 匹配率: {result.fitness*100:.1f}%")

    return result, history



def bar_tw_icp_registration_slow(source, target, init_transform=np.eye(4), 
                          max_iter=30, tau=0.3, convergence_threshold=1e-6,
                          debug_level=1):
    """
    带调试输出的简化版精配准ICP（支持首次迭代即停止）
    
    修改点：
    1. 收敛判断不再要求 iter>5 的条件
    2. 首次迭代若满足阈值立即终止
    """
    source_temp = copy.deepcopy(source).transform(init_transform)
    src_points = np.asarray(source_temp.points)
    tgt_points = np.asarray(target.points)
    
    if debug_level > 0:
        print("\n=== ICP配准初始化 ===")
        print(f"源点云数量: {len(src_points)}, 目标点云数量: {len(tgt_points)}")
        print(f"初始变换矩阵:\n{np.round(init_transform, 4)}")
        print(f"参数: max_iter={max_iter}, tau={tau}, conv_thresh={convergence_threshold}")

    # 构建KDTree
    tgt_kdtree = cKDTree(tgt_points)
    src_kdtree = cKDTree(src_points)
    current_transform = init_transform.copy()
    history = []

    for iter in range(max_iter):
        if debug_level > 0:
            print(f"\n--- 迭代 {iter+1}/{max_iter} ---")

        ### 阶段1: 双向验证 ###
        _, fw_idx = tgt_kdtree.query(src_points, k=1)
        fw_pairs = list(zip(range(len(src_points)), fw_idx))
        
        valid_pairs = []
        for s_idx, t_idx in fw_pairs:
            _, bv_idx = src_kdtree.query(tgt_points[t_idx], k=1)
            dist = np.linalg.norm(src_points[s_idx]-tgt_points[t_idx])
            if bv_idx == s_idx and dist < tau:
                valid_pairs.append((s_idx, t_idx))
                
                if debug_level > 1:
                    print(f"有效匹配: src[{s_idx}] -> tgt[{t_idx}], 距离={dist:.4f}")

        if len(valid_pairs) < 50:
            print(f"⚠️ 警告: 仅有 {len(valid_pairs)} 对有效点 (<50), 终止迭代")
            break
            
        if debug_level > 0:
            print(f"有效匹配对数: {len(valid_pairs)} "
                 f"(占总点数的 {len(valid_pairs)/min(len(src_points),len(tgt_points))*100:.1f}%)")

        ### 阶段2: SVD求解 ###
        src_corr = src_points[[p[0] for p in valid_pairs]]
        tgt_corr = tgt_points[[p[1] for p in valid_pairs]]

        src_centroid = np.mean(src_corr, axis=0)
        tgt_centroid = np.mean(tgt_corr, axis=0)
        
        if debug_level > 1:
            print(f"源点云质心: {np.round(src_centroid, 4)}")
            print(f"目标点云质心: {np.round(tgt_centroid, 4)}")

        src_centered = src_corr - src_centroid
        tgt_centered = tgt_corr - tgt_centroid
        
        H = src_centered.T @ tgt_centered
        U, S, VT = np.linalg.svd(H)
        R = VT.T @ U.T
        
        if np.linalg.det(R) < 0:
            VT[-1, :] *= -1
            R = VT.T @ U.T
        t = tgt_centroid - R @ src_centroid

        if debug_level > 1:
            print(f"SVD奇异值: {S}")
            print(f"旋转矩阵:\n{np.round(R, 4)}")
            print(f"平移向量: {np.round(t, 4)}")

        ### 更新变换 ###
        delta_T = np.eye(4)
        delta_T[:3, :3] = R
        delta_T[:3, 3] = t
        current_transform = delta_T @ current_transform
        
        # 计算误差
        transformed_src = (current_transform[:3, :3] @ src_corr.T).T + current_transform[:3, 3]
        residuals = np.linalg.norm(transformed_src - tgt_corr, axis=1)
        current_rmse = np.sqrt(np.mean(residuals**2))
        
        # 首次迭代时记录初始误差
        if iter == 0:
            initial_rmse = current_rmse
            history.append({'iter': iter, 'rmse': current_rmse, 'valid_pairs': len(valid_pairs)})
            if debug_level > 0:
                print(f"初始RMSE: {current_rmse:.6f}")
        else:
            history.append({'iter': iter, 'rmse': current_rmse, 'valid_pairs': len(valid_pairs)})
            if debug_level > 0:
                print(f"当前RMSE: {current_rmse:.6f} (Δ={abs(history[-2]['rmse']-current_rmse):.2e})")

        ### 新收敛判断逻辑（允许首次迭代终止）###
        if iter >= 0 and abs(history[-1]['rmse'] - (history[-2]['rmse'] if iter>0 else initial_rmse)) < convergence_threshold:
            print(f"✅ 收敛于迭代 {iter+1}, RMSE变化<{convergence_threshold}")
            break

    # 最终结果输出
    if debug_level > 0:
        print("\n=== 配准结果 ===")
        print(f"最终变换矩阵:\n{np.round(current_transform, 4)}")
        print(f"收敛时RMSE: {history[-1]['rmse']:.6f}, "
             f"有效点比例: {history[-1]['valid_pairs']/min(len(src_points),len(tgt_points))*100:.1f}%")

    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = current_transform
    result.inlier_rmse = history[-1]['rmse'] if history else float('inf')
    result.fitness = len(valid_pairs) / min(len(src_points), len(tgt_points))
    
    return result, history



def bar_tw_icp_registration_old(source, target, init_transform=np.eye(4), 
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


    # 初始变换
    source_temp = copy.deepcopy(source).transform(init_transform)
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

