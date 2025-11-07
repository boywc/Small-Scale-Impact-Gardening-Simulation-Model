"""
params.py —— 全局参数配置模块（单位规范 / 依赖关系 / 默认值说明）

用途
----
集中存放本项目的模拟与物理常量、统计分布及输出路径等配置，供其他模块（如 Grid、粒子追踪、
坑形叠加等）统一调用。

重要提示
--------
1) 单位约定：
   - 长度：米 m；时间：年 yr；速度：km/s；密度：kg/m^3；加速度：m/s^2；强度：Pa。
2) 数值稳定性与网格：
   - 显式扩散格式需满足 CFL 条件；隐式/Crank–Nicolson 可放宽但仍需合理 dt。
3) 文件依赖：
   - `moon_velocity_distribution.txt`：三列 [速度, 概率密度, 累积概率密度]
   - `moon_impact_sfd.txt`：两列 [尺度( km ), 累积通量 N(>D) ( km^-2·yr^-1 )]
4) 与 Grid 显式扩散的接口：
   - 若调用 `Grid.explicit_diffusion2D`，需要 `params.dx2` 与 `params.dy2`。
     通常应定义为 `dx2 = resolution**2`，`dy2 = resolution**2`
"""

import numpy as np
import os

# =========================
# 一、输入的仿真全局参数
# =========================

# 空间分辨率（每个网格像素对应的物理长度，单位 m）
resolution = 0.01  # m/px

# 时间步长 Δt（单位 year）。注意与扩散、侵蚀等过程的数值稳定性相关
dt = 1e6  # yr

# 计算域的物理宽度（单位 m）。实际网格尺寸将按 resolution 离散
grid_width = 2.0  # m

# 模型总演化时间（单位 yr）
model_time = 2e7  # yr

# ==================================================
# 二、由输入参数派生的中间量（供其它模块直接使用）
# ==================================================

# 网格像素尺度（每边的像素数，需为整数）
grid_size = int(grid_width / resolution)  # 例如 2.0/0.01 = 200 => 200×200 网格

# 仿真步数（总时间 / 时间步长）
nsteps = int(model_time / dt)

# 模拟中允许的最小撞击坑直径（用于避免分辨率以下的病态坑）
min_crater = 2.0 * resolution  # 典型取 2 个像素尺寸

# ==================================================
# 三、微陨石入射速度分布（外部文件提供统计）
# ==================================================

# 文件：三列 [v(km/s), pdf(v), cdf(v)]，cdf 单调递增至 ~1
velocity_distribution = np.loadtxt('./moon_velocity_distribution.txt')

# 速度离散点（单位 km/s）
velocities = velocity_distribution[:, 0]

# 概率密度函数 pdf(v)
velocity_probability = velocity_distribution[:, 1]

# 累积分布函数 cdf(v) = ∫ pdf dv
velocity_cumulative_probability = velocity_distribution[:, 2]

# 速度采样间隔 dv（假定为等距采样）
velocity_delta = velocities[1] - velocities[0]

# 最大速度（km/s）
velocity_max = np.max(velocities)

# 使 cdf ≈ 0.5 的速度（中位数近似；用最接近 0.5 的点）
velocity_max_likelihood = velocities[np.argmin(np.abs(velocity_cumulative_probability - 0.5))]

# 加权平均速度（以 pdf 为权重）
velocity_average = np.average(velocities, weights=velocity_probability)

# =========================
# 四、月球天体与重力常数
# =========================

# 月球半径（m）
radius_body = 1.7374e6

# 月表重力加速度（m/s^2）
g = 1.623

# =========================
# 五、撞击过程相关物性
# =========================

# 微陨石（撞击体）密度（kg/m^3）
impactor_density = 2700.0

# 风化层密度（kg/m^3）
regolith_density = 1500.0

# 基岩密度（kg/m^3），参考 Kiefer et al. 2012（玄武岩）
bedrock_density = 3148.57  # Kiefer et al., 2012

# 风化层抗剪/抗压等效强度（Pa），参考 Mitchell et al. 1972
regolith_strength = 1.0e3

# 基岩强度（Pa），参考 Asphaug et al. 1996；Marchi et al. 2009
bedrock_strength = 2.0e7

# ==========================================
# 六、微陨石尺度-频率分布 SFD（外部文件）
# ==========================================

# 文件：两列 [直径 D(km), 累积通量 N(>D) (km^-2·yr^-1)]
sfd = np.loadtxt('./moon_impact_sfd.txt')

# 尺度（km）。注意：若后续需要米制，应在使用处换算
diameter_bins_raw = sfd[:, 0]

# 累积通量 N(>D)（单位 km^-2·yr^-1）
cumulative_number_raw = sfd[:, 1]

# 默认最大陨石直径（km）。如需使用文件末值，可启用注释代码
# max_impacter_diam = diameter_bins_raw[-1]
max_impacter_diam = 0.1  # km

# =======================================
# 七、撞击入射角分布（各向同性 -> p(θ)=½ sin(2θ)）
# =======================================

# 入射角离散（0~π/2），θ=0 表示水平，θ=π/2 表示垂直
impact_angles = np.linspace(0.0, np.pi / 2.0, 1000, endpoint=True)

# 等距角步长 dθ
angle_delta = impact_angles[1] - impact_angles[0]

# 各向同性入射的解析分布：p(θ) = ½ sin(2θ)
prob = 0.5 * np.sin(2.0 * impact_angles)
prob[-1] = 0.0  # 末端点清零以避免累和超调

# 累积概率分布 cdf(θ)
cdf = np.cumsum(prob)  # 从概率密度离散和近似积分
angle_cumulative_probability = cdf / np.max(cdf)

# ============================================================
# 八、平均断裂深度与风化层厚度（随时间 ~ √t 经验标定）
# ============================================================
# 平均断裂深度和岩石厚度，使用其他地区（如高原）的已知值
# 在模型运行开始时，岩石厚度和未破裂基岩深度均为零。
# 随着运行的进行，这两层厚度随着时间 ∝ t1/2 的增加而增加。
# 在 3.5 Gyr 后产生 5 米厚的岩石层和 8 公里厚的断裂基岩层。
known_avg_fractured_depth = 10000.0  # m
known_avg_age = 4.5e9               # yr

# 参考的平均风化层厚度（m），见 Weber (Encyclopedia of the Solar System, 2014)
model_avg_regolith_thickness = 5.0  # m

# 现今假设的风化层厚度（m）
now_regolith_thickness = 2.5  # m

# 模型区域假设的平均年龄（yr）
model_avg_age = 4.5e9  # yr

# 与时间的平方根关系标定到模型区域与年龄（m）
# avg_fractured_depth ~ known_depth / sqrt(known_age) * sqrt(model_age)
avg_fractured_depth = (known_avg_fractured_depth / (known_avg_age ** 0.5)) * (model_avg_age ** 0.5)

# ====================================
# 九、连续喷出毯范围（经验系数）
# ====================================

# 连续喷出物毯的最大半径 = factor × 坑半径（典型取 5）
continuous_ejecta_blanket_factor = 5.0

# =========================
# 十、二次坑（可选）
# =========================

# 是否启用二次坑生成（0 关闭 / 1 开启）
secondaries_on = 0

# 最大二次坑直径比例：相对主坑直径的 4%
max_secondary_factor = 0.04  # 即 D_secondary_max = 0.04 * D_primary

# 二次抛射体最小离地速度（m/s），与 g 共同决定可形成二次坑的主坑下限
v_min_sec = 250.0  # m/s

# 形成二次坑的主坑最小阈值（简化近似式；单位与上下文一致）
min_primary_for_secondaries = (2.0 / 3.0) * (v_min_sec ** 2) / g

# =========================
# 十一、地形扩散（整地）系统
# =========================

# 是否开启隐式扩散（影响 Grid 初始化是否构造三对角系数）
diffusion_on = 1  # 1 开启 / 0 关闭

# 扩散系数 D（单位与方程一致；常与 dt、resolution 共同决定数值稳定性）
diffusivity = 0.109

# 备注：若使用显式格式，建议（二维均匀网格）满足 CFL：
# D·dt·(1/dx^2 + 1/dy^2) ≲ 1/2（经验），dx=dy=resolution
# 严格条件取决于离散格式，请按实际实现核对。

# =========================
# 十二、粒子追迹设置
# =========================

# 每层粒子数（用于层厚/混合深度等统计）
n_particles_per_layer = 10

# 是否启用粒子追踪（0/1）
tracers_on = 1

# 粒子是否周期性边界（1 为周期边界，0 为封闭边界）
periodic_particles = 1

# =========================
# 十三、输出目录
# =========================

# 结果保存路径（若不存在则自动创建）
save_dir = ".//Output//"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"文件夹已创建: {save_dir}")
else:
    print(f"文件夹已存在: {save_dir}")
