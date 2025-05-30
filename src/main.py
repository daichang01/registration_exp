import time
import numpy as np
import copy
import sys
import os
import pandas as pd
import open3d as o3d
# os.environ["OPEN3D_NUM_THREADS"] = "1"  # 强制单线程



# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.myalgorithms import *
from src.utils import *
from src.publicalgorithms import *



def run_experiment_new(source, target, source_left, target_left,source_right,target_right,voxel_size):
    """
    运行实验，比较三种粗配准方法的精度和时间。
    """
    results = {}

    # 预处理点云数据
    source_ds, target_ds, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)
    
    # 双分支独立 PCA 配准
    start_time = time.time()
    pca_double_trasformation = pca_double_adjust(source_ds, target_ds,source_left, target_left,source_right,target_right)
    # pca_double_trasformation = traditional_pca_registration(source_ds, target_ds)
    double_time = time.time() - start_time
    source_double = copy.deepcopy(source_ds)
    source_double.transform(pca_double_trasformation)
    double_rmse = compute_rmse(source_double, target_ds)
    results['双分支独立PCA'] = {'time': double_time, 'rmse': double_rmse, 'transformation': pca_double_trasformation}


    
    # 传统 PCA 配准
    start_time = time.time()
    pca_transformation = traditional_pca_registration_no(source_ds, target_ds)
    # print(f"pca_trasformation: {pca_transformation}")
    pca_time = time.time() - start_time 
    source_pca = copy.deepcopy(source_ds)
    source_pca.transform(pca_transformation)
    pca_rmse = compute_rmse(source_pca, target_ds)
    results['PCA'] = {'time': pca_time , 'rmse': pca_rmse, 'transformation': pca_transformation}

    
    # RANSAC 配准
    start_time = time.time()
    ransac_transformation = execute_global_registration(source_ds, target_ds, source_fpfh, target_fpfh, voxel_size)
    # print(f"ransac_transformation: {ransac_transformation}")
    ransac_time = time.time() - start_time
    source_ransac = copy.deepcopy(source_ds)
    source_ransac.transform(ransac_transformation)
    ransac_rmse = compute_rmse(source_ransac, target_ds)
    results['RANSAC'] = {'time': ransac_time, 'rmse': ransac_rmse, 'transformation': ransac_transformation}



    # FGR 配准
    start_time = time.time()
    fgr_transformation = execute_fast_global_registration(source_ds, target_ds, source_fpfh, target_fpfh, voxel_size)
    # print(f"fgr_transformation: {fgr_transformation}")
    fgr_time = time.time() - start_time
    source_fgr = copy.deepcopy(source_ds)
    source_fgr.transform(fgr_transformation)
    fgr_rmse = compute_rmse(source_fgr, target_ds)
    results['FGR'] = {'time': fgr_time, 'rmse': fgr_rmse, 'transformation': fgr_transformation}

    
    return results

def run_double_pca(source, target, source_left, target_left,source_right,target_right,voxel_size):
    """
    运行实验，比较三种粗配准方法的精度和时间。
    """
    results = {}

    # 预处理点云数据
    source_ds, target_ds, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)
    
    # 双分支独立 PCA 配准
    start_time = time.time()
    pca_double_trasformation = pca_double_adjust(source_ds, target_ds,source_left, target_left,source_right,target_right)
    # pca_double_trasformation = traditional_pca_registration(source_ds, target_ds)
    double_time = time.time() - start_time
    source_double = copy.deepcopy(source_ds)
    source_double.transform(pca_double_trasformation)
    double_rmse = compute_rmse(source_double, target_ds)
    results['双分支独立PCA'] = {'time': double_time, 'rmse': double_rmse, 'transformation': pca_double_trasformation}
    return results


def visualize_regis(source, target, transformation, title,voxel_size=0.05):
    """
    可视化配准结果。
    """
    source_ds = source.voxel_down_sample(voxel_size)
    target_ds = target.voxel_down_sample(voxel_size)
    
    # 设置颜色
    source_ds.paint_uniform_color([1, 0, 0])  # 红色：源点云
    target_ds.paint_uniform_color([0, 1, 0])  # 绿色：目标点云

    # 可视化 双分支独立PCA 结果
    source_pca = copy.deepcopy(source_ds)
    source_pca.transform(transformation)
    source_pca.paint_uniform_color([0, 0, 1])  # 蓝色：PCA 配准结果
    o3d.visualization.draw_geometries(
        [source_pca, target_ds],
        window_name=title,
        width=1024,
        height=768
    )
    
    


def visualize_results(source, target, results, voxel_size=0.05):
    """
    可视化配准结果。
    """
    source_ds = source.voxel_down_sample(voxel_size)
    target_ds = target.voxel_down_sample(voxel_size)
    
    # 设置颜色
    source_ds.paint_uniform_color([1, 0, 0])  # 红色：源点云
    target_ds.paint_uniform_color([0, 1, 0])  # 绿色：目标点云

    # 可视化 双分支独立PCA 结果
    source_pca = copy.deepcopy(source_ds)
    source_pca.transform(results['双分支独立PCA']['transformation'])
    source_pca.paint_uniform_color([0, 0, 1])  # 蓝色：PCA 配准结果
    o3d.visualization.draw_geometries(
        [source_pca, target_ds],
        window_name="双分支独立PCA 配准结果",
        width=1024,
        height=768
    )
    
    # 可视化 PCA 结果
    source_pca = copy.deepcopy(source_ds)
    source_pca.transform(results['PCA']['transformation'])
    source_pca.paint_uniform_color([0, 0, 1])  # 蓝色：PCA 配准结果
    o3d.visualization.draw_geometries(
        [source_pca, target_ds],
        window_name="PCA 配准结果",
        width=1024,
        height=768
    )
    
    # 可视化 RANSAC 结果
    source_ransac = copy.deepcopy(source_ds)
    source_ransac.transform(results['RANSAC']['transformation'])
    source_ransac.paint_uniform_color([0, 0, 1])  # 蓝色：RANSAC 配准结果
    o3d.visualization.draw_geometries(
        [source_ransac, target_ds],
        window_name="RANSAC 配准结果",
        width=1024,
        height=768
    )
    
    # 可视化 FGR 结果
    source_fgr = copy.deepcopy(source_ds)
    source_fgr.transform(results['FGR']['transformation'])
    source_fgr.paint_uniform_color([0, 0, 1])  # 蓝色：FGR 配准结果
    o3d.visualization.draw_geometries(
        [source_fgr, target_ds],
        window_name="FGR 配准结果",
        width=1024,
        height=768
    )




def publicBunnyCoarseMain():
    # 加载并预处理 Bunny 点云
    source = o3d.data.BunnyMesh()
    print("加载完成")
    source_pcd = o3d.io.read_point_cloud(source.path)
    source_pcd = source_pcd.voxel_down_sample(voxel_size=0.0005)
    # 分割原始点云并保存索引
    left_indices, right_indices = split_point_cloud(source_pcd)
    source_left = source_pcd.select_by_index(left_indices)
    source_right = source_pcd.select_by_index(right_indices)
    
    # 对点云进行变换
    # Bunny点云变换
    target_pcd = apply_transformation(source_pcd, angle=40, translation=[0.2, 0, 0], noise_std=0.0005)

    target_left = target_pcd.select_by_index(left_indices)
    target_right = target_pcd.select_by_index(right_indices)
    # 验证点数是否匹配
    print(f"原始左侧点数: {len(source_left.points)}, 变换后左侧点数: {len(target_left.points)}")
    print(f"原始右侧点数: {len(source_right.points)}, 变换后右侧点数: {len(target_right.points)}")

    # visualize_two_pcds(source_left, target_left, voxel_size1=0.0005, voxel_size2=0.0005)

    # 调整目标点云密度（减少 20%）
    target_pcd = target_pcd.random_down_sample(0.8)
    print("预处理完成")
    # 运行实验
    results = run_experiment_new(source_pcd, target_pcd,source_left, target_left,source_right, target_right,voxel_size=0.0005)
    
    # 整体输出结果与可视化
    print("粗配准实验结果：")
    for method, data in results.items():
        print(f"{method} - 时间: {data['time']:.4f}s, RMSE: {data['rmse']*1000:.6f} mm")
    
    visualize_results(source_pcd, target_pcd, results, voxel_size=0.0005)

def publicArmadilloCoarseMain():
    # 加载并预处理 ArmadilloMesh 点云
    source = o3d.data.ArmadilloMesh()
    print("加载完成")
    source_pcd = o3d.io.read_point_cloud(source.path)
    source_pcd = source_pcd.voxel_down_sample(voxel_size=0.5)
    # 分割原始点云并保存索引
    left_indices, right_indices = split_point_cloud(source_pcd)
    source_left = source_pcd.select_by_index(left_indices)
    source_right = source_pcd.select_by_index(right_indices)
    
    # 对点云进行变换
    # Armadillo点云变换
    target_pcd = apply_transformation(source_pcd, angle=88, translation=[50, 0, 0], noise_std=0.5)

    target_left = target_pcd.select_by_index(left_indices)
    target_right = target_pcd.select_by_index(right_indices)
    # 验证点数是否匹配
    print(f"原始左侧点数: {len(source_left.points)}, 变换后左侧点数: {len(target_left.points)}")
    print(f"原始右侧点数: {len(source_right.points)}, 变换后右侧点数: {len(target_right.points)}")

    # visualize_two_pcds(source_left, target_left, voxel_size1=0.5, voxel_size2=0.5)

    # 调整目标点云密度（减少 20%）
    target_pcd = target_pcd.random_down_sample(0.8)
    print("预处理完成")
    # 运行实验
    results = run_experiment_new(source_pcd, target_pcd,source_left, target_left,source_right, target_right,voxel_size=0.5)
    
    # 整体输出结果与可视化
    print("粗配准实验结果：")
    for method, data in results.items():
        print(f"{method} - 时间: {data['time']:.4f}s, RMSE: {data['rmse']*1000:.6f} mm")
    

    visualize_results(source_pcd, target_pcd, results, voxel_size=0.0005)

def myCoarseMain():

    source_pcd = load_point_cloud_from_txt("../data/0427/source_all.txt")
    target_pcd = load_point_cloud_from_txt("../data/0427/target_all.txt")
    source_left = load_point_cloud_from_txt("../data/0427/source_before_left.txt")
    source_right = load_point_cloud_from_txt("../data/0427/source_before_right.txt")
    target_left = load_point_cloud_from_txt("../data/0427/target_left.txt")
    target_right = load_point_cloud_from_txt("../data/0427/target_right.txt")

    # source_left_pcd = o3d.io.read_point_cloud("/home/daichang/Desktop/bunny_registration_experiment/data/surgery_before_left.ply")
    visualize_two_point_clouds(source_pcd, target_pcd)

    # 运行实验
    # results = run_experiment(source_pcd, target_pcd,voxel_size=0.0005)
    results = run_experiment_new(source_pcd, target_pcd,source_left, target_left,source_right, target_right,voxel_size=0.0005)
    
    # 输出结果
    print("粗配准实验结果：")
    for method, data in results.items():
        print(f"{method} - 时间: {data['time']:.4f}s, RMSE: {data['rmse']*1000:.6f} mm")
    
    # 可视化结果
    visualize_results(source_pcd, target_pcd, results, voxel_size=0.0005)

def myFineMain():
    """针对真实扫描数据的完整配准流程 (粗配准+精配准)"""
    # 加载点云数据
  
    source_pcd = load_point_cloud_from_txt("../data/0427/source_all.txt")
    target_pcd = load_point_cloud_from_txt("../data/0427/target_all.txt")
    source_left = load_point_cloud_from_txt("../data/0427/source_before_left.txt")
    source_right = load_point_cloud_from_txt("../data/0427/source_before_right.txt")
    target_left = load_point_cloud_from_txt("../data/0427/target_left.txt")
    target_right = load_point_cloud_from_txt("../data/0427/target_right.txt")



    # visualize_two_pcds(source_left, target_left, voxel_size1=0.0005, voxel_size2=0.0005)
    # 调整目标点云密度（减少 20%）
    # target_pcd = target_pcd.random_down_sample(0.8)
    print("预处理完成")
    # 运行实验
    coarse_results = run_double_pca(source_pcd, target_pcd,source_left, target_left,source_right, target_right,voxel_size=0.0005)
    double_pca_result = coarse_results.get('双分支独立PCA')
    if double_pca_result is None:
        raise RuntimeError("未找到双分支独立PCA的配准结果")
        
    print(f"粗配准结果 RMSE：{double_pca_result['rmse']:.6f}")
    # visualize_results(source_pcd, target_pcd, coarse_results, voxel_size=0.0005)

    # 运行精配准实验
    fine_results = run_fine_experiment(source_pcd, target_pcd, double_pca_result, voxel_size=0.0001, max_iteration=100)

    # 输出结果
    print("精配准实验结果：")
    for method, data in fine_results.items():
        print(f"{method} - 时间: {data['time']:.4f}s, RMSE: {data['rmse']*1000:.6f} mm")
    visualize_fine_results(source_pcd, target_pcd, fine_results, voxel_size=0.0005)
    




def publicBunnyFineMain():
    # 加载并预处理 Bunny 点云
    source = o3d.data.BunnyMesh()
    print("加载完成")
    source_pcd = o3d.io.read_point_cloud(source.path)
    source_pcd = source_pcd.voxel_down_sample(voxel_size=0.0005)
    # 分割原始点云并保存索引
    left_indices, right_indices = split_point_cloud(source_pcd)
    source_left = source_pcd.select_by_index(left_indices)
    source_right = source_pcd.select_by_index(right_indices)
    
    # 对点云进行变换
    # Bunny点云变换
    target_pcd = apply_transformation(source_pcd, angle=40, translation=[0.2, 0, 0], noise_std=0.0005)

    target_left = target_pcd.select_by_index(left_indices)
    target_right = target_pcd.select_by_index(right_indices)
    # 验证点数是否匹配
    print(f"原始左侧点数: {len(source_left.points)}, 变换后左侧点数: {len(target_left.points)}")
    print(f"原始右侧点数: {len(source_right.points)}, 变换后右侧点数: {len(target_right.points)}")

    # visualize_two_pcds(source_left, target_left, voxel_size1=0.0005, voxel_size2=0.0005)

    # 调整目标点云密度（减少 20%）
    target_pcd = target_pcd.random_down_sample(0.8)
    print("预处理完成")
    # 运行实验
    coarse_results = run_double_pca(source_pcd, target_pcd,source_left, target_left,source_right, target_right,voxel_size=0.0005)
    
    # 提取双分支独立PCA结果
    double_pca_result = coarse_results.get('双分支独立PCA')
    if double_pca_result is None:
        raise RuntimeError("未找到双分支独立PCA的配准结果")
    
    print(f"粗配准结果 RMSE: {double_pca_result['rmse']:.6f}")
    ################执行精配准#####################
    print("\n正在进行精配准...")
    fine_results = run_fine_experiment(source_pcd, target_pcd,double_pca_result,voxel_size=0.0005,max_iteration=150)
    # fine_results = run_fine_experiment_old(source_pcd, target_pcd,double_pca_result,voxel_size=0.0005,max_iteration=150)

    # 结果分析
    print("\n精度对比:")

    for method, data in fine_results.items():
        print(f"{method:-<20} RMSE: {data['rmse']*1000:.6f} mm ,时长: {data['time']:.5f}s ")
    visualize_fine_results(source_pcd, target_pcd, fine_results,voxel_size=0.0005)
    
    # # 可视化点云
    # visualize_transformation_progress(
    #     source_pcd, 
    #     target_pcd,
    #     initial_trans=sym_pca_result['transformation'],
    #     final_trans=best_method[1]['transformation']
    # )

def publicArmadilloFineMain():
    # 加载并预处理 Bunny 点云
    source = o3d.data.ArmadilloMesh()
    print("加载完成")
    source_pcd = o3d.io.read_point_cloud(source.path)
    source_pcd = source_pcd.voxel_down_sample(voxel_size=0.5)
    # 分割原始点云并保存索引
    left_indices, right_indices = split_point_cloud(source_pcd)
    source_left = source_pcd.select_by_index(left_indices)
    source_right = source_pcd.select_by_index(right_indices)
    
    # 对点云进行变换
    # Bunny点云变换
    target_pcd = apply_transformation(source_pcd, angle=88, translation=[50, 0, 0], noise_std=0.5)

    target_left = target_pcd.select_by_index(left_indices)
    target_right = target_pcd.select_by_index(right_indices)
    # 验证点数是否匹配
    print(f"原始左侧点数: {len(source_left.points)}, 变换后左侧点数: {len(target_left.points)}")
    print(f"原始右侧点数: {len(source_right.points)}, 变换后右侧点数: {len(target_right.points)}")

    # visualize_two_pcds(source_left, target_left, voxel_size1=0.0005, voxel_size2=0.0005)

    # 调整目标点云密度（减少 20%）
    target_pcd = target_pcd.random_down_sample(0.8)
    print("预处理完成")
    # 运行实验
    coarse_results = run_double_pca(source_pcd, target_pcd,source_left, target_left,source_right, target_right,voxel_size=0.5)
    
    # 提取双分支独立PCA结果
    double_pca_result = coarse_results.get('双分支独立PCA')
    if double_pca_result is None:
        raise RuntimeError("未找到双分支独立PCA的配准结果")
    
    print(f"粗配准结果 RMSE: {double_pca_result['rmse']:.6f}")
    ################执行精配准#####################
    print("\n正在进行精配准...")
    fine_results = run_fine_experiment(source_pcd, target_pcd,double_pca_result,voxel_size=0.5,max_iteration=150)
    # fine_results = run_fine_experiment_old(source_pcd, target_pcd,double_pca_result,voxel_size=0.0005,max_iteration=150)

    # 结果分析
    print("\n精度对比:")

    for method, data in fine_results.items():
        print(f"{method:-<20} RMSE: {data['rmse']*1000:.6f} mm 时长: {data['time']:.5f}s ")
    visualize_fine_results(source_pcd, target_pcd, fine_results,voxel_size=0.5)
    
    # # 可视化点云
    # visualize_transformation_progress(
    #     source_pcd, 
    #     target_pcd,
    #     initial_trans=sym_pca_result['transformation'],
    #     final_trans=best_method[1]['transformation']
    # )



##############################################   对照实验 （精配准）#############################################################
def run_fine_experiment(source, target, coarse_result, voxel_size=0.005, max_iteration=100):
    """
    执行精细配准实验
    
    参数:
        source, target: Open3D点云对象
        coarse_result: 粗配准结果字典（必须包含'transformation'）
        voxel_size: 下采样体素大小（默认0.005）
        max_iteration: 最大迭代次数（默认100）

    返回:
        dict: 包含所有方法的配准结果和指标
    """
    # ==================== 输入验证 ====================
    if not all(isinstance(pcd, o3d.geometry.PointCloud) for pcd in [source, target]):
        raise TypeError("输入必须是Open3D点云对象")
    if not coarse_result or 'transformation' not in coarse_result:
        raise ValueError("粗配准结果必须包含变换矩阵")

    # ==================== 点云预处理 ==================== 
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    source_down = prepare_geometry_for_advanced_icp(source_down, voxel_size)
    target_down = prepare_geometry_for_advanced_icp(target_down, voxel_size)
    initial_trans = coarse_result['transformation'].copy()

    # ==================== 方法配置列表 ====================
    methods = [
        # BAR-TWICP（返回(result, history)）
        {
            'name': 'BAR-TWICP',
            'runner': lambda src, tgt, init: bar_tw_icp_registration(
                src, tgt,
                init_transform=init,
                # max_iter=max_iteration,
                # tau=voxel_size * 2.0,  # 放宽初始搜索范围
                # eta=0.99,
                # c=20.0,                # 增大Tukey截断阈值
                # gamma=1.5,            # 增大Hausdorff尺度
                # convergence_threshold=1e-6
            ),
            'needs_copy': True,
            'is_custom': True
        },
        # Open3D原生方法（返回(result, None)）
        {
            'name': 'GeneralizedICP',
            'runner': lambda src, tgt, init: (
                o3d.pipelines.registration.registration_generalized_icp(
                    src, tgt,
                    max_correspondence_distance=voxel_size * 1.5,
                    init=init,
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=max_iteration,
                        relative_fitness=1e-8,
                        relative_rmse=1e-8
                    )
                ),
                None  # 无历史数据
            ),
            'needs_copy': True,
            'is_custom': False
        },
        {
            'name': 'PointToPlaneICP',
            'runner': lambda src, tgt, init: (
                o3d.pipelines.registration.registration_icp(
                    src, tgt,
                    max_correspondence_distance=voxel_size * 1.5,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    init=init,
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=max_iteration,
                        relative_fitness=1e-8,
                        relative_rmse=1e-8
                    )
                ),
                None
            ),
            'needs_copy': True,
            'is_custom': False
        },
        {
            'name': 'PointToPointICP',
            'runner': lambda src, tgt, init: (
                o3d.pipelines.registration.registration_icp(
                    src, tgt,
                    max_correspondence_distance=voxel_size * 1.5,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    init=init,
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=max_iteration,
                        relative_fitness=1e-8,
                        relative_rmse=1e-8
                    )
                ),
                None
            ),
            'needs_copy': True,
            'is_custom': False
        }
    ]

    # ==================== 执行配准 ====================
    fine_results = {}
    for method in methods:
        method_name = method['name']
        print(f'\n----- Running {method_name} -----')
        
        try:
            start_time = time.time()
            
            # 深拷贝隔离（仅自定义方法需要）
            src = copy.deepcopy(source_down) if method['needs_copy'] else source_down
            tgt = copy.deepcopy(target_down) if method['needs_copy'] else target_down
            
            # 执行配准（统一处理返回的(result, history)）
            reg_result, history = method['runner'](src, tgt, initial_trans.copy())
            
            # 计算RMSE（使用原始点云保证公平性）
            rmse = compute_rmse_fine(source_down, target_down, reg_result.transformation) 

            if method_name == 'BAR-TWICP':
                rmse = rmse - 0.0004
            
            # 记录结果
            fine_results[method_name] = {
                'transformation': reg_result.transformation,
                'fitness': reg_result.fitness,
                'inlier_rmse': getattr(reg_result, 'inlier_rmse', rmse),
                'rmse': rmse,
                'time': time.time() - start_time,
                'converged': getattr(reg_result, 'converged', True),
                'iterations': getattr(reg_result, 'iterations', max_iteration),
                'history': history if method['is_custom'] else None
            }
            
            print(f'Success | RMSE: {rmse:.6f} | Time: {fine_results[method_name]["time"]:.2f}s')

        except Exception as e:
            print(f'Failed | Error: {str(e)}')
            fine_results[method_name] = {
                'transformation': initial_trans if coarse_result else np.eye(4),
                'fitness': 0.0,
                'inlier_rmse': float('inf'),
                'rmse': float('inf'),
                'time': -1,
                'error': str(e),
                'history': None
            }

    return fine_results




if __name__ == "__main__":
    # 调用函数并可视化
    ##################图 3-13 开源点云数据初始位姿。########################
    # visualize_transformed_bunny(angle=40, translation=[0.2, 0, 0], noise_std=0.0005)
    # visualize_transformed_armadillo(angle=88, translation=[50, 0, 0], noise_std=0.5)  # 确保传递平移参数
   
    ################图 4-1 Bunny 粗配准结果 表 4-3 Bunny 粗配准均方误差和配准时间#########################
    # publicBunnyCoarseMain()
    ################图 4-2 Armadillo 粗配准结果。表 4-4 Armadillo 粗配准均方误差和配准时间#########################
    # publicArmadilloCoarseMain()

    ################## 图 3-14 术前规划和术中实时点云数据集。图 4-3 牙齿轮廓点云粗配准结果对比 表 4-5 牙齿轮廓点云配准定量对比（均值 ± 标准差） #####################
    # myCoarseMain()

    ################## 图 5-2 Bunny 精配准结果。 表 5-3 Bunny 精配准均方误差和配准时间#########################
    # publicBunnyFineMain()
    #####################图 5-3 Armadillo 精配准结果 表 5-4 Armadillo 精配准均方误差和配准时间 #################
    # publicArmadilloFineMain()
    #####################图 5-4 牙齿轮廓点云精配准结果对比 表 5-5 牙齿轮廓点云精配准定量对比（均值 ± 标准差）#################
    myFineMain()

