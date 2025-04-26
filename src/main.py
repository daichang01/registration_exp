import time
import numpy as np
import copy
import sys
import os
import pandas as pd
import open3d as o3d

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
    
    # 对称双分支 PCA 配准
    start_time = time.time()
    pca_double_trasformation = pca_double_adjust(source_ds, target_ds,source_left, target_left,source_right,target_right)
    # pca_double_trasformation = traditional_pca_registration(source_ds, target_ds)
    double_time = time.time() - start_time
    source_double = copy.deepcopy(source_ds)
    source_double.transform(pca_double_trasformation)
    double_rmse = compute_rmse(source_double, target_ds)
    results['对称双分支PCA'] = {'time': double_time, 'rmse': double_rmse, 'transformation': pca_double_trasformation}


    
    # 传统 PCA 配准
    start_time = time.time()
    pca_transformation = traditional_pca_registration_no(source_ds, target_ds)
    # print(f"pca_trasformation: {pca_transformation}")
    pca_time = time.time() - start_time
    source_pca = copy.deepcopy(source_ds)
    source_pca.transform(pca_transformation)
    pca_rmse = compute_rmse(source_pca, target_ds)
    results['PCA'] = {'time': pca_time, 'rmse': pca_rmse, 'transformation': pca_transformation}

    
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


def run_experiment(source, target, voxel_size=0.05):
    """
    运行实验，比较三种粗配准方法的精度和时间。
    """
    results = {}

    # 预处理点云数据
    source_ds, target_ds, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)
    

    # 对称双分支 PCA 配准
    start_time = time.time()
    pca_transformation = traditional_pca_registration(source_ds, target_ds)
    pca_time = time.time() - start_time
    source_pca = copy.deepcopy(source_ds)
    source_pca.transform(pca_transformation)
    pca_rmse = compute_rmse(source_pca, target_ds)
    results['对称双分支PCA'] = {'time': pca_time, 'rmse': pca_rmse, 'transformation': pca_transformation}
    
    # 传统 PCA 配准
    start_time = time.time()
    pca_transformation = traditional_pca_registration_no(source_ds, target_ds)
    pca_time = time.time() - start_time
    source_pca = copy.deepcopy(source_ds)
    source_pca.transform(pca_transformation)
    pca_rmse = compute_rmse(source_pca, target_ds)
    results['PCA'] = {'time': pca_time, 'rmse': pca_rmse, 'transformation': pca_transformation}
    
    # RANSAC 配准
    start_time = time.time()
    ransac_transformation = execute_global_registration(source_ds, target_ds, source_fpfh, target_fpfh, voxel_size)
    ransac_time = time.time() - start_time
    source_ransac = copy.deepcopy(source_ds)
    source_ransac.transform(ransac_transformation)
    ransac_rmse = compute_rmse(source_ransac, target_ds)
    results['RANSAC'] = {'time': ransac_time, 'rmse': ransac_rmse, 'transformation': ransac_transformation}
    
    # FGR 配准
    start_time = time.time()
    fgr_transformation = execute_fast_global_registration(source_ds, target_ds, source_fpfh, target_fpfh, voxel_size)
    fgr_time = time.time() - start_time
    source_fgr = copy.deepcopy(source_ds)
    source_fgr.transform(fgr_transformation)
    fgr_rmse = compute_rmse(source_fgr, target_ds)
    results['FGR'] = {'time': fgr_time, 'rmse': fgr_rmse, 'transformation': fgr_transformation}
    
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

    # 可视化 对称双分支PCA 结果
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

    # 可视化 对称双分支PCA 结果
    source_pca = copy.deepcopy(source_ds)
    source_pca.transform(results['对称双分支PCA']['transformation'])
    source_pca.paint_uniform_color([0, 0, 1])  # 蓝色：PCA 配准结果
    o3d.visualization.draw_geometries(
        [source_pca, target_ds],
        window_name="对称双分支PCA 配准结果",
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




def publicCoarseMain():
        # 加载并预处理 Bunny 点云
    source = o3d.data.BunnyMesh()
    # source = o3d.data.ArmadilloMesh()
    print("加载完成")
    source_pcd = o3d.io.read_point_cloud(source.path)
    source_pcd = source_pcd.voxel_down_sample(voxel_size=0.0005)
    # 分割原始点云并保存索引
    left_indices, right_indices = split_point_cloud(source_pcd)
    source_left = source_pcd.select_by_index(left_indices)
    source_right = source_pcd.select_by_index(right_indices)
    
    # 对点云进行变换
    target_pcd = apply_transformation(source_pcd, angle=40, translation=[0.2, 0, 0], noise_std=0.0005)

    target_left = target_pcd.select_by_index(left_indices)
    target_right = target_pcd.select_by_index(right_indices)
    # 验证点数是否匹配
    print(f"原始左侧点数: {len(source_left.points)}, 变换后左侧点数: {len(target_left.points)}")
    print(f"原始右侧点数: {len(source_right.points)}, 变换后右侧点数: {len(target_right.points)}")

    visualize_two_pcds(source_left, target_left, voxel_size1=0.0005, voxel_size2=0.0005)



    #     # 调整目标点云密度（减少 20%）
    # target_pcd = target_pcd.random_down_sample(0.8)
    # visualize_two_pcds(source_pcd, target_pcd, voxel_size1=0.0005, voxel_size2=0.0005)

    print("预处理完成")
    # 运行实验
    # results = run_experiment(source_pcd, target_pcd,voxel_size=0.0005)
    results = run_experiment_new(source_pcd, target_pcd,source_left, target_left,source_right, target_right,voxel_size=0.0005)
    
    # 整体输出结果与可视化
    print("粗配准实验结果：")
    for method, data in results.items():
        print(f"{method} - 时间: {data['time']:.4f}s, RMSE: {data['rmse']:.6f}")
    

    visualize_results(source_pcd, target_pcd, results, voxel_size=0.0005)

def mymain():

    source_pcd = load_point_cloud_from_txt("../data/before_0307_161339.txt")
    target_pcd = load_point_cloud_from_txt("../data/real_0307_161339.txt")
    visualize_two_point_clouds(source_pcd, target_pcd)

    # 运行实验
    results = run_experiment(source_pcd, target_pcd,voxel_size=0.0005)
    
    # 输出结果
    print("粗配准实验结果：")
    for method, data in results.items():
        print(f"{method} - 时间: {data['time']:.4f}s, RMSE: {data['rmse']:.6f}")
    
    # 可视化结果
    visualize_results(source_pcd, target_pcd, results, voxel_size=0.0005)

def myfinemain():
    """针对真实扫描数据的完整配准流程 (粗配准+精配准)"""
    # 加载点云数据
    try:
        source_pcd = load_point_cloud_from_txt("../data/before_0307_161339.txt")
        target_pcd = load_point_cloud_from_txt("../data/real_0307_161339.txt")
        print("点云加载成功，原始点云数量:")
        print(f"源点云: {len(source_pcd.points)} 点 | 目标点云: {len(target_pcd.points)} 点")
    except Exception as e:
        print(f"点云加载失败: {str(e)}")
        return

    # 初始终始可视化对比
    print("\n正在显示初始点云对比...")
    visualize_two_point_clouds(source_pcd, target_pcd)

    # 运行粗配准实验
    print("\n正在进行粗配准...")
    try:
        # 根据数据特性设置下采样参数（可根据实际点云密度调整）
        voxel_size = 0.0005  
        coarse_results = run_experiment(source_pcd, target_pcd, voxel_size=voxel_size)
        
         # 提取双分支PCA结果
        sym_pca_result = coarse_results.get('对称双分支PCA')
        if sym_pca_result is None:
            raise RuntimeError("未找到对称双分支PCA的配准结果")
        # ====== 新增：粗配准结果可视化 ======
        print("\n正在显示粗配准后效果对比...")
        # 应用最优变换矩阵到源点云（创建副本避免修改原始数据）
        transformed_source = copy.deepcopy(source_pcd)
        transformed_source.transform(sym_pca_result['transformation'])
        # 复用现有可视化函数
        visualize_two_point_clouds(
            transformed_source,  # 变换后的源点云
            target_pcd         # 原始目标点云

        )
        # 输出粗配准结果
        print("\n粗配准结果对比:")
        for method, data in coarse_results.items():
            print(f"{method:-<15} | 耗时: {data['time']:.3f}s | RMSE: {data['rmse']:.6f}")
    except Exception as e:
        print(f"粗配准失败: {str(e)}")
        return

    # 执行精配准
    print("\n正在进行精配准...")
    try:
        # 根据数据密度调整精配准参数
        fine_voxel_size = 0.0002  # 更精细的下采样
        max_iterations = 100     # 最大迭代次数
        
        fine_results = run_fine_experiment_single(
            source_pcd, 
            target_pcd,
            sym_pca_result,
            voxel_size=fine_voxel_size,
            max_iteration=max_iterations
        )

        # 结果分析
        print("\n精配准性能对比:")
        print(f"初始粗配准RMSE: {sym_pca_result['rmse']:.6f}")
        for method, data in fine_results.items():
            print(f"{method:-<15} | 耗时: {data['time']:.6f}s | Fitness: {data['fitness']:.6f} | RMSE: {data['rmse']:.6f}")

        # 可视化最佳结果
        best_method = max(fine_results.items(), key=lambda x: x[1]['fitness'])
        print(f"\n最佳精配准方法: {best_method[0]} (配准度: {best_method[1]['fitness']:.3f})")

        # 可视化所有精配准结果
        visualize_fine_results(source_pcd, target_pcd, fine_results, voxel_size=fine_voxel_size)

        # 可选：显示配准过程动画
        # visualize_transformation_progress(
        #     source_pcd, 
        #     target_pcd,
        #     initial_trans=sym_pca_result['transformation'],
        #     final_trans=best_method[1]['transformation']
        # )
        
    except Exception as e:
        print(f"精配准失败: {str(e)}")
        return



def publicfinemain():
    # 加载并预处理点云
    source = o3d.data.ArmadilloMesh()
    # source = o3d.data.BunnyMesh()
    source_pcd = o3d.io.read_point_cloud(source.path)
    source_pcd = source_pcd.voxel_down_sample(voxel_size=0.5)
    
    # 生成目标点云 (与参数设置匹配)
    target_pcd = apply_transformation(
        source_pcd, 
        angle=40, 
        translation=[20, 0, 0],
        noise_std=0.01  # 适当增大噪声值
    )

    # 调整目标点云密度（减少 20%）
    # current_num_points = len(target_pcd.points)
    # target_num_points = int(current_num_points * 0.3)  # 减少 20%
    # target_pcd = target_pcd.random_down_sample(target_num_points / current_num_points)
    # print(f"原始点云数量: {current_num_points}")
    # print(f"降采样后点云数量: {len(target_pcd.points)}")
    # 运行粗配准实验
    print("\n正在进行粗配准...")
    coarse_results = run_experiment(source_pcd, target_pcd, voxel_size=0.5)
    # 提取对称双分支PCA结果
    sym_pca_result = coarse_results.get('对称双分支PCA')
    if sym_pca_result is None:
        raise RuntimeError("未找到对称双分支PCA的配准结果")
    

    
    # 执行精配准
    print("\n正在进行精配准...")
    fine_results = run_fine_experiment_single(
        source_pcd, 
        target_pcd,
        sym_pca_result,
        voxel_size=0.5,
        max_iteration=150
    )
    # 结果分析
    print("\n精度对比:")
    print(f"粗配准结果 RMSE: {sym_pca_result['rmse']:.6f}")
    for method, data in fine_results.items():
        print(f"{method:-<20} RMSE: {data['rmse']:.6f} 时长: {data['time']:.5f}s 配准度: {data['fitness']:.3f}")
    # 可视化最佳结果
    best_method = max(fine_results.items(), key=lambda x: x[1]['fitness'])
    print(f"\n最佳精配准方法: {best_method[0]} (配准度: {best_method[1]['fitness']:.3f})")

    visualize_fine_results(source_pcd, target_pcd, fine_results)
    
    # # 可视化点云
    # visualize_transformation_progress(
    #     source_pcd, 
    #     target_pcd,
    #     initial_trans=sym_pca_result['transformation'],
    #     final_trans=best_method[1]['transformation']
    # )







if __name__ == "__main__":
    # 调用函数并可视化
    ##################初始姿态  图3-13a 3-13b########################
    visualize_transformed_bunny(angle=40, translation=[0.2, 0, 0], noise_std=0.0)
    visualize_transformed_armadillo(angle=40, translation=[50, 0, 0], noise_std=0.0)  # 确保传递平移参数
   
    ################ 开源粗配准#########################
    publicCoarseMain()
    ################## 自建数据粗配准 #####################
    # mymain()
    # 开源精配准
    # publicfinemain()
    # 自建数据精配准
    # myfinemain()

     ##################工具函数########################
    # save_for_cloudcompare()
    # source_process("/home/daichang/Desktop/bunny_registration_experiment/data/source.ply","/home/daichang/Desktop/bunny_registration_experiment/data")
    # save_for_doublepcd("/home/daichang/Desktop/bunny_registration_experiment/data/source-right.pcd")
    # ply_to_pcd("/home/daichang/Desktop/bunny_registration_experiment/data/source - right.ply","/home/daichang/Desktop/bunny_registration_experiment/data","source-right.pcd")
    # pcd_to_ply("/home/daichang/Desktop/bunny_registration_experiment/data/target_right.pcd","/home/daichang/Desktop/bunny_registration_experiment/data","target-right.ply")