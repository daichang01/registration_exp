import time
import numpy as np
import copy
import sys
import os
import pandas as pd
import open3d as o3d

from src.utils import *

#############################################   对照实验 （粗配准）#############################################################

def traditional_pca_registration(source, target):
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

def traditional_pca_registration_no(source, target):
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

    # 计算旋转矩阵（不进行特征向量方向校准）
    R = target_eigvecs @ source_eigvecs.T

    # 计算平移向量
    t = target_centroid - R @ source_centroid

    # 构造变换矩阵
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t

    # time.sleep(1.7)

    return transformation



def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """基于 RANSAC 的全局配准"""
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result.transformation


def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """基于 FGR 的快速全局配准"""
    distance_threshold = voxel_size * 0.5
    # print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result.transformation



##############################################   对照实验 （精配准）#############################################################
def run_fine_experiment_single(source, target, coarse_result, voxel_size=0.005, max_iteration=100):
    """
    修复版：基于单个粗配准结果执行精配准实验
    """
    # 参数检查
    if coarse_result is None:
        raise ValueError("必须提供粗配准结果")
    # 下采样与法线计算
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    # 确保法线和协方差的存在 (关键修复点)
    source_down = prepare_geometry_for_advanced_icp(source_down, voxel_size)
    target_down = prepare_geometry_for_advanced_icp(target_down, voxel_size)
    # 结果字典初始化
    fine_results = {}
    initial_trans = coarse_result['transformation']
    # Generalized ICP 参数配置（需要协方差）
    gen_icp_config = {
        'method': 'GeneralizedICP',
        'registration_func': o3d.pipelines.registration.registration_generalized_icp,
        'params': {
            'criteria': o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iteration,
                relative_fitness=1e-8,
                relative_rmse=1e-8
            )
        }
    }
    # 其他ICP方法配置
    methods = [
        {
            'method': 'PointToPlaneICP',
            'registration_func': o3d.pipelines.registration.registration_icp,
            'params': {
                'estimation_method': o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                'criteria': o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=max_iteration,
                    relative_fitness=1e-8,
                    relative_rmse=1e-8
                )
            }
        },
        {
            'method': 'PointToPointICP',
            'registration_func': o3d.pipelines.registration.registration_icp,
            'params': {
                'estimation_method': o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
                'criteria': o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=max_iteration,
                    relative_fitness=1e-8,
                    relative_rmse=1e-8
                )
            }
        }
    ]
    # 注册Generalized ICP需要单独处理首个方法
    try:
        start_time = time.time()
        gen_icp_result = gen_icp_config['registration_func'](
            source_down, target_down,
            max_correspondence_distance=voxel_size*1.5,
            init=initial_trans,
            **gen_icp_config['params']
        )
        # 验证结果有效性 (新增检查点)
        if not validate_registration_result(gen_icp_result):
            raise RuntimeError("Generalized ICP 返回无效结果")
        
        fine_results[gen_icp_config['method']] = build_result_data(gen_icp_result, compute_rmse, start_time, source_down, target_down)
    except Exception as e:
        print(f"Generalized ICP 严重错误: {str(e)}")
        fine_results[gen_icp_config['method']] = error_fallback_data(initial_trans)
    # 处理其他ICP方法
    for method in methods:
        try:
            start_time = time.time()
            result = method['registration_func'](
                source_down, target_down,
                max_correspondence_distance=voxel_size*1.5,
                init=initial_trans,
                **method['params']
            )
            # 有效性检查
            if not validate_registration_result(result):
                raise RuntimeError(f"{method['method']} 返回无效结果")
                
            fine_results[method['method']] = build_result_data(result, compute_rmse, start_time, source_down, target_down)
        except Exception as e:
            print(f"{method['method']} 失败: {str(e)}")
            fine_results[method['method']] = error_fallback_data(initial_trans)
    return fine_results

def prepare_geometry_for_advanced_icp(pcd, voxel_size):
    """修复版本：协方差估计"""
    # 确保法线存在
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size*2, max_nn=30))
    
    # 关键修复点：修改方法名为正确的estimate_covariances
    if not pcd.has_covariances():
        pcd.estimate_covariances(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(  # 参数格式调整
                radius=voxel_size*3,
                max_nn=50
            )
        )
    return pcd

def run_generalized_icp(source, target, init_trans, max_iter=100, fitness=1e-6, rmse=1e-6):
    """执行 Generalized ICP 精配准"""
    try:
        # 使用 registration_generalized_icp
        result = o3d.pipelines.registration.registration_generalized_icp(
            source, target, max_correspondence_distance=0.1, init=init_trans,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter,
                relative_fitness=fitness,
                relative_rmse=rmse)
        )
        return result.transformation
    except Exception as e:
        print(f"Generalized ICP 错误: {e}")
        return init_trans  # 如果失败，返回初始变换矩阵

def run_point_to_plane_icp(source, target, init_trans, max_iter=100, fitness=1e-6, rmse=1e-6):
    """执行 Point-to-Plane ICP 精配准"""
    return o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=0.1, init=init_trans,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter,
            relative_fitness=fitness,
            relative_rmse=rmse)
    ).transformation

def run_point_to_point_icp(source, target, init_trans, max_iter=100, fitness=1e-6, rmse=1e-6):
    """执行 Point-to-Point ICP 精配准"""
    return o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=0.1, init=init_trans,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter,
            relative_fitness=fitness,
            relative_rmse=rmse)
    ).transformation

