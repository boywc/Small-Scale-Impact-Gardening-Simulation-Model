import numpy as np
import matplotlib.pyplot as plt


class SphereVoxel(object):

    def __init__(self, position):
        self.position = position
        self.exposure_age = 0

    def get_position(self):
        return self.position

    def refresh_position(self, new_position):
        self.position = new_position

    def get_exposure_age(self):
        return self.exposure_age

    def set_exposure_age(self, add_age):
        self.exposure_age += add_age


def generate_sphere_group(n_points, radius=1.0):
    """生成面元体素组"""
    group = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # 黄金角度

    for i in range(n_points):
        y = 1 - (i / (n_points - 1)) * 2  # y从1到-1
        radius_at_y = np.sqrt(1 - y * y)  # 在y处的半径

        theta = phi * i  # 黄金角度增量

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y

        group.append(SphereVoxel([x * radius, y * radius, z * radius]))

    return group


def random_rotation_matrix():
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

    return Rz @ Ry @ Rx


def rotate_point_cloud(group):
    rotation_matrix = random_rotation_matrix()
    points = np.array([i.get_position() for i in group])
    rotated_points = np.dot(points, rotation_matrix.T)
    for i, cla in enumerate(group):
        cla.refresh_position(rotated_points[i])


def grain_exposure(group, add_age, h_thro=0):
    for i in group:
        if i.get_position()[2] > h_thro:
            i.set_exposure_age(add_age)

def grain_exposure_consider_self(group, add_age):
    for i in group:
        if i.get_position()[2] > 0:
            sintheta = i.get_position()[2] / 1
            i.set_exposure_age(add_age * sintheta)

def get_average_exposure_age(group):
    exposure_age_list = [i.get_exposure_age() for i in group]
    return np.mean(exposure_age_list)


def plot_colored_point_cloud(group):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = np.array([i.get_exposure_age() for i in group])
    points = np.array([i.get_position() for i in group])
    sc = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors, cmap='viridis', s=5, alpha=0.8
    )

    # 添加颜色条
    plt.colorbar(sc, ax=ax, label='Exposure Age')
    # ax.set_title('A')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    plt.show()


def display_paper_image():
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

    ###  2. 绘制粒子强度分布
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
    ax2.scatter([0], [0], [0], c="r", s=10, alpha=0.8)
    # 绘制1/4园面 x
    r = np.linspace(0, 1, 50)
    theta = np.linspace(0, np.pi / 2, 50)
    R, Theta = np.meshgrid(r, theta)
    Y = R * np.cos(Theta)
    Z = R * np.sin(Theta)
    X = np.zeros_like(Y)  # x=0
    ax2.plot_surface(X, Y, Z, color='gray', alpha=1, edgecolor='none', antialiased=False, shade=False)
    # 绘制1/4园面 y
    r = np.linspace(0, 1, 50)
    theta = np.linspace(0, np.pi / 2, 50)
    R, Theta = np.meshgrid(r, theta)
    Y = np.zeros_like(Y)
    Z = R * np.sin(Theta)
    X = R * np.cos(Theta)
    ax2.plot_surface(X, Y, Z, color='gray', alpha=1)
    # 绘制1/4园面 z
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

    ###  3. 绘制粒子暴露两次
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

    ###  4. 绘制粒子暴露100次
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



    # 绘制水平面





    # ax.set_title('A')




if __name__ == "__main__":
    display_paper_image()

    group = generate_sphere_group(1000)
    add_age = 0.1
    real_exposure_age = []
    record_exposure_age = []
    for i in range(100):
        rotate_point_cloud(group)
        # grain_exposure(group, add_age, h_thro=0.0)
        grain_exposure_consider_self(group, add_age)
        real_exposure_age.append(i*add_age)
        record_exposure_age.append(get_average_exposure_age(group))
        # plot_colored_point_cloud(group)
        if i ==0 or i == 2 or i == 3 or i == 30 or i==60 or i==99:
            plot_colored_point_cloud(group)
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    # ax.set_title('球面点云')
    plt.plot(real_exposure_age, record_exposure_age)
    plt.xlabel('Exposure Time (Myr)')
    plt.ylabel('Record Age (Myr)')
    plt.show()
