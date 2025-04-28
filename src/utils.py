import time
import numpy as np
import copy
import sys
import os
import pandas as pd
import open3d as o3d
from scipy.spatial import cKDTree

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def visualize_transformed_bunny(angle=30, translation=[0.5, 0, 0], noise_std=0.0):
    """
    正确版本：显示原始点云（红色）和变换后点云（蓝色）
    """
    # 加载原始点云
    source = o3d.data.BunnyMesh()
    source_pcd = o3d.io.read_point_cloud(source.path)
    source_pcd = source_pcd.voxel_down_sample(voxel_size=0.003)
    
    # 创建变换后的点云（不影响原点云）
    target_pcd = apply_transformation( # 返回新的点云对象
        source_pcd, 
        angle=angle, 
        translation=translation,
        noise_std=noise_std
    )
    
    # 设置颜色
    source_pcd.paint_uniform_color([1, 0, 0])  # 红色
    target_pcd.paint_uniform_color([0, 0, 1])  # 蓝色
    
    # 添加坐标系
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    # 显示两个点云
    o3d.visualization.draw_geometries(
        [source_pcd, target_pcd],
        window_name="变换前后点云对比",
        width=1024,
        height=768
    )


def visualize_transformed_armadillo(angle=30, translation=[0.5, 0, 0], noise_std=0.0):
    """
    正确版本：显示原始点云（红色）和变换后点云（蓝色）
    """
    # 加载原始点云
    source = o3d.data.ArmadilloMesh()
    source_pcd = o3d.io.read_point_cloud(source.path)
    source_pcd = source_pcd.voxel_down_sample(voxel_size=0.01)
    
    # 创建变换后的点云（不影响原点云）
    target_pcd = apply_transformation( # 返回新的点云对象
        source_pcd, 
        angle=angle, 
        translation=translation,  # 确保正确传递平移参数
        noise_std=noise_std
    )
    
    # 设置颜色
    source_pcd.paint_uniform_color([1, 0, 0])  # 红色
    target_pcd.paint_uniform_color([0, 0, 1])  # 蓝色
    
    # 显示两个点云
    o3d.visualization.draw_geometries(
        [source_pcd, target_pcd],
        window_name="变换前后点云对比",
        width=1024,
        height=768
    )

def source_process(source_path="/home/daichang/Desktop/bunny_registration_experiment/data/source.ply",save_dir="/home/daichang/Desktop/bunny_registration_experiment/data"):
    source_pcd = o3d.io.read_point_cloud(source_path)
    source_pcd = apply_transformation(source_pcd, angle=0, translation=[0, 0, 0], noise_std=0.0005)
    current_num_points = len(source_pcd.points)
    target_num_points = int(current_num_points * 0.8)  # 减少 20%
    souce_pcd = source_pcd.random_down_sample(target_num_points / current_num_points)

    os.makedirs(save_dir, exist_ok=True)  # 确保文件夹存在

    # 保存为 PLY 格式（CloudCompare 默认支持）
    # o3d.io.write_point_cloud(os.path.join(save_dir, "source.ply"), source_pcd)
    o3d.io.write_point_cloud(os.path.join(save_dir, "source_process.ply"), source_pcd)



def save_for_doublepcd(source_path="/home/daichang/Desktop/bunny_registration_experiment/data/source-left.pcd",save_dir="/home/daichang/Desktop/bunny_registration_experiment/data"):
    source_pcd = o3d.io.read_point_cloud(source_path)
    
    # 对点云进行变换
    target_pcd = apply_transformation(source_pcd, angle=40, translation=[0.2, 0, 0], noise_std=0.0005)
    
    #     # 调整目标点云密度（减少 20%）
    current_num_points = len(target_pcd.points)
    target_num_points = int(current_num_points * 0.8)  # 减少 20%
    target_pcd = target_pcd.random_down_sample(target_num_points / current_num_points)

    os.makedirs(save_dir, exist_ok=True)  # 确保文件夹存在

    # 保存为 PLY 格式（CloudCompare 默认支持）
    # o3d.io.write_point_cloud(os.path.join(save_dir, "source.ply"), source_pcd)
    o3d.io.write_point_cloud(os.path.join(save_dir, "target_right.pcd"), target_pcd)

    # 也可保存为 PCD 或 XYZ 格式（任选其一）
    # o3d.io.write_point_cloud(os.path.join(save_dir, "source.pcd"), source_pcd)
    # o3d.io.write_point_cloud(os.path.join(save_dir, "target.pcd"), target_pcd)

    # o3d.io.write_point_cloud(os.path.join(save_dir, "source.xyz"), source_pcd)
    # o3d.io.write_point_cloud(os.path.join(save_dir, "target.xyz"), target_pcd)

    print(f"点云已保存至 {save_dir}/")


def save_for_cloudcompare(save_dir="/home/daichang/Desktop/bunny_registration_experiment/data"):
    """
    保存点云以供 CloudCompare 使用
    Args:
        source_pcd: 源点云（open3d.geometry.PointCloud）
        target_pcd: 目标点云（open3d.geometry.PointCloud）
        save_dir: 保存目录
    """
    source = o3d.data.BunnyMesh()
    # source = o3d.data.ArmadilloMesh()
    print("加载完成")
    source_pcd = o3d.io.read_point_cloud(source.path)
    source_pcd = source_pcd.voxel_down_sample(voxel_size=0.0005)
    
    # 对点云进行变换
    target_pcd = apply_transformation(source_pcd, angle=40, translation=[0.2, 0, 0], noise_std=0.0005)
    
    #     # 调整目标点云密度（减少 20%）
    current_num_points = len(target_pcd.points)
    target_num_points = int(current_num_points * 0.8)  # 减少 20%
    target_pcd = target_pcd.random_down_sample(target_num_points / current_num_points)

    os.makedirs(save_dir, exist_ok=True)  # 确保文件夹存在

    # 保存为 PLY 格式（CloudCompare 默认支持）
    o3d.io.write_point_cloud(os.path.join(save_dir, "source.ply"), source_pcd)
    o3d.io.write_point_cloud(os.path.join(save_dir, "target.ply"), target_pcd)

    # 也可保存为 PCD 或 XYZ 格式（任选其一）
    # o3d.io.write_point_cloud(os.path.join(save_dir, "source.pcd"), source_pcd)
    # o3d.io.write_point_cloud(os.path.join(save_dir, "target.pcd"), target_pcd)

    # o3d.io.write_point_cloud(os.path.join(save_dir, "source.xyz"), source_pcd)
    # o3d.io.write_point_cloud(os.path.join(save_dir, "target.xyz"), target_pcd)

    print(f"点云已保存至 {save_dir}/")


def ply_to_pcd(ply_path, pcd_save_dir, output_filename="converted.pcd"):
    """
    将PLY文件转换为PCD格式并保存到指定路径
    
    参数：
        ply_path (str): PLY文件的输入路径（如 "input.ply"）
        pcd_save_dir (str): PCD文件的保存目录（如 "C:/output"）
        output_filename (str): 输出的PCD文件名（默认 "converted.pcd"）
    """
    # 创建保存目录（如果不存在）
    os.makedirs(pcd_save_dir, exist_ok=True)
    # 1. 读取PLY文件
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_points():
        raise ValueError("PLY文件无有效点云数据！")
    # 2. 构造输出路径
    pcd_path = os.path.join(pcd_save_dir, output_filename)
    # 3. 保存为PCD格式
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"✅ 转换完成：PLY → PCD\n保存路径：{pcd_path}")



def pcd_to_ply(pcd_path, ply_save_dir, output_filename="converted.ply"):
    """
    将PCD文件转换为PLY格式并保存到指定路径
    
    参数：
        pcd_path (str): PCD文件的输入路径（如 "input.pcd"）
        ply_save_dir (str): PLY文件的保存目录（如 "C:/output"）
        output_filename (str): 输出的PLY文件名（默认 "converted.ply"）
    """
    # 创建保存目录（如果不存在）
    os.makedirs(ply_save_dir, exist_ok=True)

    # 1. 读取PCD文件
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_points():
        raise ValueError("PCD文件无有效点云数据！")

    # 2. 构造输出路径
    ply_path = os.path.join(ply_save_dir, output_filename)

    # 3. 保存为PLY格式
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"✅ 转换完成：PCD → PLY\n保存路径：{ply_path}")

def split_point_cloud(pcd):
    points = np.asarray(pcd.points)
    left_mask = points[:, 0] < 0   # X坐标小于0的点（左侧）
    right_mask = ~left_mask        # 剩余点（右侧）
    left_indices = np.where(left_mask)[0]  # 左侧点的索引
    right_indices = np.where(right_mask)[0] # 右侧点的索引
    return left_indices, right_indices

def visualize_two_pcds(
    pcd1: o3d.geometry.PointCloud,
    pcd2: o3d.geometry.PointCloud,
    voxel_size1=0.01,
    voxel_size2=0.01,
    color1=[1, 0, 0],  # 默认红色
    color2=[0, 1, 0],  # 默认绿色
    window_name="Two Point Clouds",
    merge=True,
    background_color=[0, 0, 0],  # 默认黑色背景
    point_size=2.0,
):
    """
    可视化两个点云，支持设置不同的 voxel_size 和颜色。

    Args:
        pcd1 (o3d.geometry.PointCloud): 第一个点云
        pcd2 (o3d.geometry.PointCloud): 第二个点云
        voxel_size1 (float): 第一个点云的体素大小（若<=0 则不降采样）
        voxel_size2 (float): 第二个点云的体素大小（若<=0 则不降采样）
        color1 (list): 第一个点云颜色 [r, g, b], 范围0~1
        color2 (list): 第二个点云颜色 [r, g, b], 范围0~1
        window_name (str): 窗口名称
        merge (bool): True 合并后可视化，False 分开可视化
        background_color (list): 背景颜色 [r, g, b]
        point_size (float): 点的大小
    """
    # 降采样点云
    if voxel_size1 > 0:
        pcd1 = pcd1.voxel_down_sample(voxel_size1)
    if voxel_size2 > 0:
        pcd2 = pcd2.voxel_down_sample(voxel_size2)
    
    # 设置颜色
    pcd1.paint_uniform_color(color1)
    pcd2.paint_uniform_color(color2)

    # 合并或分别可视化
    if merge:
        merged_pcd = pcd1 + pcd2
        vis_geometries = [merged_pcd]
    else:
        vis_geometries = [pcd1, pcd2]

    # 构建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)

    # 添加点云
    for geom in vis_geometries:
        vis.add_geometry(geom)

    # 设置渲染选项
    render_opt = vis.get_render_option()
    render_opt.background_color = np.asarray(background_color)
    render_opt.point_size = point_size

    # 运行可视化
    vis.run()
    vis.destroy_window()


###################################################### 辅助工具函数 ########################################################

def apply_transformation(source_pcd, angle=30, translation=[0.5, 0, 0], noise_std=0.0):
    """
    对点云进行旋转、平移和噪声处理（返回新点云，不修改原始点云）。
    """
    # 创建副本确保原始点云不被修改
    transformed_pcd = copy.deepcopy(source_pcd)
    
    # 计算质心和主轴方向
    points = np.asarray(transformed_pcd.points)
    centroid = np.mean(points, axis=0)
    cov_matrix = np.cov(points - centroid, rowvar=False)
    eigen_vals, eigen_vecs = np.linalg.eigh(cov_matrix)
    main_axis = eigen_vecs[:, np.argmax(eigen_vals)]  # 最大特征值对应的主方向

    # 构建旋转变换
    theta = np.radians(angle)
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(main_axis * theta)
    transformed_pcd.rotate(R, center=centroid)  # 作用在副本上

    # 构建平移变换
    transformed_pcd.translate(translation)  # 作用在副本上

    # 添加噪声
    if noise_std > 0:
        points = np.asarray(transformed_pcd.points)
        noise = np.random.normal(0, noise_std, size=points.shape)
        transformed_pcd.points = o3d.utility.Vector3dVector(points + noise)

    return transformed_pcd  # 返回新的变换点云
def prepare_dataset(source, target, voxel_size):
    """特征提取预处理
    Args:
        source (o3d.geometry.PointCloud): 源点云数据
        target (o3d.geometry.PointCloud): 目标点云数据
        voxel_size (float): 体素下采样的体素大小
    Returns:
        tuple: 包含下采样后的源点云、目标点云及其FPFH特征
    """
    
    # 对源点云进行体素下采样，减少点云数量
    source_ds = source.voxel_down_sample(voxel_size)
    # 对目标点云进行体素下采样，减少点云数量
    target_ds = target.voxel_down_sample(voxel_size)
    
    # 设置法线估计的搜索半径（通常为体素大小的5倍）
    radius = voxel_size * 5
    
    # 估计源点云的法线（使用混合KD树搜索策略，半径搜索+最近邻数量限制）
    source_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius, 64))
    # 估计目标点云的法线（同上）
    target_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius, 64))
    
    # 计算源点云的FPFH特征（Fast Point Feature Histogram）
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_ds, 
        o3d.geometry.KDTreeSearchParamHybrid(radius, 100))  # 使用更大的最近邻数量100
    
    # 计算目标点云的FPFH特征
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_ds, 
        o3d.geometry.KDTreeSearchParamHybrid(radius, 100))
    
    # 返回处理后的数据：下采样点云 + FPFH特征
    return source_ds, target_ds, source_fpfh, target_fpfh


def compute_rmse(source, target):
    """精度评估"""
    dists = source.compute_point_cloud_distance(target)
    return np.sqrt(np.mean(np.square(dists)))

# def compute_rmse_fine_old(source_points, target_points, trans_matrix):
#     """计算全局RMSE（考虑所有对应点）"""
#     transformed = np.dot(source_points, trans_matrix[:3, :3].T) + trans_matrix[:3, 3]
#     tree = cKDTree(target_points)
#     distances, _ = tree.query(transformed, k=1)
#     valid_mask = ~np.isinf(distances)
#     return np.sqrt(np.mean(distances[valid_mask]**2))

def compute_rmse_fine(source, traget, trans_matrix):
    """计算全局RMSE（考虑所有对应点）"""
    transformed_source = copy.deepcopy(source)
    transformed_source.transform(trans_matrix)
    dists = transformed_source.compute_point_cloud_distance(traget)
    return np.sqrt(np.mean(np.square(dists)))


def crop_cloud(cloud, ratio=0.9, axis=None):
    """
    对点云进行裁剪，保留指定比例的部分。

    参数：
    - cloud: 输入的点云对象。
    - ratio: 裁剪比例，表示保留的部分占整体的比例（默认 0.9）。
    - axis: 裁剪方向（如果为 None，则使用点云的主轴方向）。

    返回：
    - 裁剪后的点云对象。
    """
    # 获取点云的点坐标
    points = np.asarray(cloud.points)

    # 如果未指定裁剪方向，使用点云的主轴方向
    if axis is None:
        # 计算点云的主轴方向
        centroid = np.mean(points, axis=0)
        cov_matrix = np.cov(points - centroid, rowvar=False)
        eigen_vals, eigen_vecs = np.linalg.eigh(cov_matrix)
        axis = eigen_vecs[:, np.argmax(eigen_vals)]  # 最大特征值对应的方向为裁剪方向

    # 将点投影到裁剪方向上
    proj = np.dot(points - centroid, axis)

    # 根据裁剪比例确定保留的区域
    threshold = np.percentile(proj, (1 - ratio) * 100)  # 保留投影值较大的部分
    mask = proj >= threshold

    # 根据掩码选择点
    cropped_cloud = cloud.select_by_index(np.where(mask)[0])

    return cropped_cloud

def load_point_cloud_from_txt(file_path):
    """
    从txt文件加载点云数据并转换为Open3D点云对象
    :param file_path: txt文件路径
    :return: Open3D点云对象
    """
    # 从txt文件加载数据为numpy数组
    points = np.loadtxt(file_path)
    
    # 检查数据是否正确加载
    if points.size == 0:
        raise ValueError("Point cloud data is empty.")
    
    # 创建Open3D点云对象
    point_cloud = o3d.geometry.PointCloud()
    
    # 将numpy数组转换为Open3D点的格式
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    return point_cloud

def visualize_two_point_clouds(source_pcd, target_pcd):
    """
    在同一个视图中可视化两个点云
    :param source_pcd: 第一个点云对象（Open3D.PointCloud）
    :param target_pcd: 第二个点云对象（Open3D.PointCloud）
    """
    # 设置点云颜色（可选）
    source_pcd.paint_uniform_color([0, 0, 1])  # 红色表示source_pcd
    target_pcd.paint_uniform_color([0, 1, 0])  # 绿色表示target_pcd

    # 可视化两个点云
    o3d.visualization.draw_geometries([source_pcd, target_pcd])

def visualize_transformation_progress(source, target, initial_trans, final_trans):
    """可视化配准过程"""
    # 初始化显示配置
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='配准过程演示', width=1200, height=900)
    
    # 创建点云副本
    source_orig = copy.deepcopy(source).paint_uniform_color([1, 0, 0])    # 红色：原始点云
    source_coarse = copy.deepcopy(source).transform(initial_trans).paint_uniform_color([0, 0, 1])  # 蓝色：粗配准
    source_fine = copy.deepcopy(source).transform(final_trans).paint_uniform_color([0, 1, 0])      # 绿色：精配准
    target_display = copy.deepcopy(target).paint_uniform_color([0.5, 0.5, 0.5])  # 灰色：目标点云
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    # 分批添加几何体实现渐变动画效果
    vis.add_geometry(source_orig)
    vis.add_geometry(target_display)
    vis.add_geometry(coordinate_frame)
    
    # 渐入粗配准结果
    for i in range(10):
        vis.add_geometry(source_coarse.transform(np.eye(4)), reset_bounding_box=False)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)
    
    # 渐入精配准结果
    for i in range(20):
        vis.add_geometry(source_fine.transform(np.eye(4)), reset_bounding_box=False)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.03)
    
    # 保持窗口不关闭
    vis.run()
    vis.destroy_window()

def validate_registration_result(result):
    """ 验证配准结果有效性 """
    if not hasattr(result, 'transformation'):
        return False
    # 检查变换矩阵合理性
    return np.isfinite(result.transformation).all()
def build_result_data(result, rmse_func, start_time, source, target):
    """ 构建结果数据结构 """
    transformed_source = copy.deepcopy(source)
    transformed_source.transform(result.transformation)
    return {
        'time': time.time() - start_time,
        'rmse': rmse_func(transformed_source, target),
        'fitness': result.fitness,
        'transformation': result.transformation
    }
def error_fallback_data(initial_trans):
    """ 错误返回默认数据 """
    return {
        'time': -1,
        'rmse': -1,
        'fitness': -1,
        'transformation': initial_trans  # 保留初始变换用于可视化
    }



def visualize_fine_results(source, target, fine_results, voxel_size=0.005):
    """精配准可视化"""
    # 预处理保持原逻辑
    source_ds = source.voxel_down_sample(voxel_size)
    target_ds = target.voxel_down_sample(voxel_size)
    
    colors = {
        "source": [1, 0, 0],     # 红: 源点云初始位置
        "result": [0, 0.5, 1],   # 蓝: 配准结果
        "target": [0.2, 0.8, 0.2]  # 绿: 目标点云
    }
    
    # 基准展示元素（与粗配准保持一致）
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    for method_name, result in fine_results.items():
        # 忽略无效结果
        if result.get('invalid', False):
            continue
            
        # 克隆并变换源点云
        aligned_source = copy.deepcopy(source_ds)
        aligned_source.transform(result['transformation'])
        
        # 构建显示对象列表
        geometries = [
            _colorize_cloud(target_ds, colors["target"]),
            _colorize_cloud(aligned_source, colors["result"])
            # coordinate_frame
        ]
        
        # 窗口标题携带主要指标
        title = f"精配准: {method_name} | "
        title += f"RMSE:{result['rmse']:.6f}"
        

        # 启动可视化
        o3d.visualization.draw_geometries(
            geometries,
            window_name=title,
            width=1024,
            height=768
        )

def _colorize_cloud(cloud, color):
    """安全颜色转换函数"""
    clone = copy.deepcopy(cloud)
    clone.paint_uniform_color(color)
    return clone