import numpy as np
import matplotlib.pyplot as plt


class SphereVoxel(object):
    """
    表示球面上一“体素/面元”的最小数据单元。

    每个 SphereVoxel 持有其在球面上的 3D 坐标与累计“暴露龄”（exposure_age）。
    在本模型中，球体半径默认 1（单位化），位置更新与暴露龄累积通过旋转与条件累加实现。

    Attributes:
        position (list[float]): 三维位置 [x, y, z]，位于单位球面上（|r|≈1）。
        exposure_age (float): 该面元累计“暴露龄”（无量纲或外部指定单位，如 Myr/次等）。
    """

    def __init__(self, position):
        self.position = position
        self.exposure_age = 0

    def get_position(self):
        """返回当前 3D 位置坐标。"""
        return self.position

    def refresh_position(self, new_position):
        """用新的 3D 坐标更新该面元位置（就地更新）。"""
        self.position = new_position

    def get_exposure_age(self):
        """返回当前累计暴露龄。"""
        return self.exposure_age

    def set_exposure_age(self, add_age):
        """在当前暴露龄上累加 add_age（加性更新）。"""
        self.exposure_age += add_age


def generate_sphere_group(n_points, radius=1.0):
    """
    生成均匀分布在球面上的面元组（Fibonacci / 黄金角采样）。

    采用“黄金角度”分布在球面上生成 n_points 个采样点，再封装为 SphereVoxel。
    该方法能较好逼近球面上的均匀采样，适合可视化与统计。

    Args:
        n_points (int): 采样点/面元数量。
        radius (float): 球半径，默认 1.0。

    Returns:
        list[SphereVoxel]: 面元对象列表，每个元素带有初始位置与零暴露龄。
    """
    """生成面元体素组"""
    group = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # 黄金角度（Fibonacci sphere 的关键常数）

    for i in range(n_points):
        # 将 y 由 1 均匀插值到 -1，相当于“纬向”均匀
        y = 1 - (i / (n_points - 1)) * 2  # y从1到-1
        radius_at_y = np.sqrt(1 - y * y)  # 在该 y 截面的圆半径（单位球上）

        theta = phi * i  # 沿经向以黄金角递增，避免簇拥

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y

        group.append(SphereVoxel([x * radius, y * radius, z * radius]))

    return group


def random_rotation_matrix():
    """
    随机生成一个 3D 旋转矩阵（XYZ 欧拉角的依次旋转组合）。

    做法：分别在 [0, 2π) 为 x/y/z 轴采样随机旋转角，构造 Rx/Ry/Rz，
    然后以 R = Rz @ Ry @ Rx 组合返回。

    Returns:
        np.ndarray: 形状 [3,3] 的旋转矩阵，满足正交与 det(R)=+1（数值误差内）。
    """
    """生成一个随机的3D旋转矩阵"""
    theta_x = np.random.uniform(0, 2 * np.pi)
    theta_y = np.random.uniform(0, 2 * np.pi)
    theta_z = np.random.uniform(0, 2 * np.pi)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])

    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    # 组合顺序：先绕 x，再绕 y，最后绕 z
    return Rz @ Ry @ Rx


def rotate_point_cloud(group):
    """
    对整个面元组做一次整体“刚体旋转”（就地更新每个面元的位置）。

    注：该旋转不改变球面半径，仅改变点在球面上的方位分布，便于模拟“随机取向”的
    多次暴露过程。

    Args:
        group (list[SphereVoxel]): 面元对象集合。
    """
    rotation_matrix = random_rotation_matrix()
    points = np.array([i.get_position() for i in group])  # shape=[N,3]
    rotated_points = np.dot(points, rotation_matrix.T)    # 右乘 R^T 等价于左乘 R
    for i, cla in enumerate(group):
        cla.refresh_position(rotated_points[i])           # 就地更新


def grain_exposure(group, add_age, h_thro=0):
    """
    简化“可见/暴露”判定：当面元 z 分量高于阈值 h_thro，累计 add_age。

    该规则可理解为：以 z=h_thro 为“地平/遮挡面”，其上的面元被认为处于“暴露状态”，
    每次调用累计一次等量的暴露龄。

    Args:
        group (list[SphereVoxel]): 面元集合。
        add_age (float): 每次满足条件时的等量增量。
        h_thro (float): z 阈值，默认 0（以赤道为界）。
    """
    for i in group:
        if i.get_position()[2] > h_thro:
            i.set_exposure_age(add_age)


def grain_exposure_consider_self(group, add_age):
    """
    自遮蔽权重版本：以面元法向 z 分量作为权重，使“朝向上半球”的面元累加更多暴露龄。

    思想：
        - 在单位球上，法向对“上方半空间”的暴露几率与 z 分量（即 sin(theta)）相关。
        - 这里使用 sintheta = z/1（半径为 1）作为权重，使 z 越大（越朝上），增量越多。

    与项目背景的关系：
        - 在对球面进行“多次随机取向+按朝向计权”的长期累积后，平均记录的暴露龄将低于
          实际（表面）暴露时间；在你项目的可视化说明中采用了“除以 4”的经验修正因子，
          即“记录暴露龄 ≈ 实际表面停留时间的 1/4”，该经验来自本 3D 粒子模拟。:contentReference[oaicite:2]{index=2}

    Args:
        group (list[SphereVoxel]): 面元集合。
        add_age (float): 基础增量（会再乘以权重 z）。
    """
    for i in group:
        if i.get_position()[2] > 0:
            sintheta = i.get_position()[2] / 1  # 单位球半径=1 => 权重≈z
            i.set_exposure_age(add_age * sintheta)


def get_average_exposure_age(group):
    """
    计算面元集合的平均暴露龄（便于与“真实暴露时长”对比做校准/回归）。

    Args:
        group (list[SphereVoxel]): 面元集合。

    Returns:
        float: 平均暴露龄。
    """
    exposure_age_list = [i.get_exposure_age() for i in group]
    return np.mean(exposure_age_list)


def plot_colored_point_cloud(group):
    """
    按暴露龄着色绘制 3D 球面散点图（颜色表示 exposure_age）。

    可用于：
        - 观察一次/多次曝光后的空间分布与累积差异
        - 与“可视化/论文配图”对齐（色条展示）

    Args:
        group (list[SphereVoxel]): 面元集合。
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = np.array([i.get_exposure_age() for i in group])
    points = np.array([i.get_position() for i in group])

    # 散点着色：暴露龄越大颜色越“热”
    sc = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors, cmap='viridis', s=5, alpha=0.8
    )

    # 颜色条（标注 Exposure Age）
    plt.colorbar(sc, ax=ax, label='Exposure Age')

    # 轴设定与等比例显示
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    plt.show()


def display_paper_image():
    """
    生成 2×2 图板，演示“单次暴露/两次暴露/百次暴露”的累计效果与象限剖视。

    面板含义：
        1) 全貌：随机旋转+按自遮蔽权重累加后的粒子分布与着色
        2) 强度剖视：留 1/8 空间（x>0,y>0,z>0）便于观察内部结构，并绘制三个 1/4 圆面
        3) 暴露两次：在 (1) 基础上再随机旋转并累计一次
        4) 暴露一百次：继续迭代 98 次，展示长期累积效果（用于直观说明“均值低估”）

    该图板与项目说明中“暴露龄记录低于实际表面停留时间，经验采用×1/4 修正”的论述
    一致：多次随机取向并按朝向加权会使平均记录值低于真实值。:contentReference[oaicite:3]{index=3}
    """
    group = generate_sphere_group(1000)
    rotate_point_cloud(group)
    grain_exposure_consider_self(group, 1)

    fig = plt.figure(figsize=(10, 8))

    ###  1. 绘制粒子全貌
    ax1 = fig.add_subplot(221, projection='3d')
    colors = np.array([i.get_exposure_age() for i in group])
    points = np.array([i.get_position() for i in group])
    sc = ax1.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors, cmap='viridis', s=2, alpha=0.8
    )
    # 灰色水平面（z=0）：相当于“地平/遮挡面”参考
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    ax1.plot_surface(X, Y, Z, color='gray', alpha=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=30, azim=45)
    ax1.set_box_aspect([1, 1, 1])

    ###  2. 绘制粒子强度分布（剖视：仅保留 x>0,y>0,z>0 八分体）
    ax2 = fig.add_subplot(222, projection='3d')
    colors = np.array([i.get_exposure_age() for i in group])
    points = np.array([i.get_position() for i in group])
    remove_mask_x = points[:, 0] > 0
    remove_mask_y = points[:, 1] > 0
    remove_mask_z = points[:, 2] > 0
    remove_mask = ~(remove_mask_x * remove_mask_y * remove_mask_z)
    points2 = points[remove_mask]
    colors2 = colors[remove_mask]
    sc = ax2.scatter(
        points2[:, 0], points2[:, 1], points2[:, 2],
        c=colors2, cmap='viridis', s=2, alpha=1
    )
    ax2.scatter([0], [0], [0], c="r", s=10, alpha=0.8)  # 原点

    # 绘制三个 1/4 圆面，帮助理解坐标轴与剖视
    # 绘制1/4圆面 x（x=0）
    r = np.linspace(0, 1, 50)
    theta = np.linspace(0, np.pi / 2, 50)
    R, Theta = np.meshgrid(r, theta)
    Y = R * np.cos(Theta)
    Z = R * np.sin(Theta)
    X = np.zeros_like(Y)  # x=0
    ax2.plot_surface(X, Y, Z, color='gray', alpha=1, edgecolor='none', antialiased=False, shade=False)

    # 绘制1/4圆面 y（y=0）
    r = np.linspace(0, 1, 50)
    theta = np.linspace(0, np.pi / 2, 50)
    R, Theta = np.meshgrid(r, theta)
    Y = np.zeros_like(Y)
    Z = R * np.sin(Theta)
    X = R * np.cos(Theta)
    ax2.plot_surface(X, Y, Z, color='gray', alpha=1)

    # 绘制1/4圆面 z（z=0）
    r = np.linspace(0, 1, 50)
    theta = np.linspace(0, np.pi / 2, 50)
    R, Theta = np.meshgrid(r, theta)
    Y = R * np.sin(Theta)
    Z = np.zeros_like(Y)
    X = R * np.cos(Theta)
    ax2.plot_surface(X, Y, Z, color='gray', alpha=1)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_box_aspect([1, 1, 1])
    ax2.view_init(elev=30, azim=45)

    ###  3. 绘制粒子暴露两次（再随机旋转一次并累加）
    ax3 = fig.add_subplot(223, projection='3d')
    rotate_point_cloud(group)
    grain_exposure_consider_self(group, 1)
    colors = np.array([i.get_exposure_age() for i in group])
    points = np.array([i.get_position() for i in group])
    sc = ax3.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors, cmap='viridis', s=2, alpha=0.8
    )
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_box_aspect([1, 1, 1])
    ax3.view_init(elev=30, azim=45)

    ###  4. 绘制粒子暴露100次（长期累积效果）
    ax4 = fig.add_subplot(224, projection='3d')
    for i in range(98):
        rotate_point_cloud(group)
        grain_exposure_consider_self(group, 1)
    colors = np.array([i.get_exposure_age() for i in group])
    points = np.array([i.get_position() for i in group])
    sc = ax4.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors, cmap='viridis', s=2, alpha=0.8
    )
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_box_aspect([1, 1, 1])
    ax4.view_init(elev=30, azim=45)

    plt.show()

    # 绘制水平面（上方已在子图中覆盖）


    # ax.set_title('A')


if __name__ == "__main__":
    # 论文图板演示：2×2 面板（单次/两次/百次暴露等）
    display_paper_image()

    # 下方：构造“真实暴露时间 vs. 记录均值”的关系曲线，用于直观观察“低估效应”
    group = generate_sphere_group(1000)
    add_age = 0.1
    real_exposure_age = []    # 理想“真实暴露时长”（线性累加）
    record_exposure_age = []  # 粒子集合“记录的平均暴露龄”（会低于真实值）
    for i in range(100):
        rotate_point_cloud(group)
        # grain_exposure(group, add_age, h_thro=0.0)  # 等权版本：仅基于阈值
        grain_exposure_consider_self(group, add_age)   # 自遮蔽权重版本：更贴近实际
        real_exposure_age.append(i*add_age)
        record_exposure_age.append(get_average_exposure_age(group))
        # plot_colored_point_cloud(group)
        if i ==0 or i == 2 or i == 3 or i == 30 or i==60 or i==99:
            plot_colored_point_cloud(group)

    # 对比曲线：如需与项目说明中“×1/4 修正”呼应，可据此拟合或标注。:contentReference[oaicite:4]{index=4}
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    # ax.set_title('球面点云')
    plt.plot(real_exposure_age, record_exposure_age)
    plt.xlabel('Exposure Time (Myr)')
    plt.ylabel('Record Age (Myr)')
    plt.show()
