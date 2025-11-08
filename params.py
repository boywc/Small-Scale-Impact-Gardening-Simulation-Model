import numpy as np
import os
# 输入的仿真参数
resolution = 0.01 # 分辨率 单位m
dt = 1e6 # 时间步长 单位year
grid_width = 2.0 # 网格对应真实宽度 单位m
model_time = 2e7 # 模型仿真总时间 单位year

# 根据仿真参数计算得到的中间参数
grid_size = int(grid_width/resolution) # 网格像素尺度
nsteps = int(model_time/dt) # 仿真运行次数
min_crater = 2.0*resolution # 仿真中最小撞击坑的直径

# 微陨石速度参数
velocity_distribution = np.loadtxt('./moon_velocity_distribution.txt')
velocities = velocity_distribution[:, 0]                       # 第一列是速度
velocity_probability = velocity_distribution[:, 1]             # 第二列是概率密度
velocity_cumulative_probability = velocity_distribution[:, 2]  # 第三列是累积概率密度
velocity_delta = velocities[1] - velocities[0]                 # 每一组的速度间隔 dv
velocity_max = np.max(velocities)                              # km/s # 速度最大值
velocity_max_likelihood = velocities[np.argmin(np.abs(velocity_cumulative_probability - 0.5))]  # 累积概率密度为 0.5的速度
velocity_average = np.average(velocities, weights=velocity_probability)                         # 速度的加权平均数

# 月球参数
radius_body = 1.7374e6  # 月球半径 单位m
g = 1.623               # 月球重力加速度 单位 m/s^2

# 撞击过程参数
impactor_density = 2700.0  # 微陨石密度 kg/m^3
regolith_density = 1500.0  # 风化层密度 kg/m^3
bedrock_density = 3148.57  # 月球玄武岩密度 kg/m^3 Kiefer et al. 2012
regolith_strength = 1.0e3  # 岩石强度 Pa, from Mitchell et al. 1972
bedrock_strength = 2.0e7   # 基岩强度 Pa, from Marchi et al. 2009 via Asphaug et al. 1996

# 微陨石尺度频率分布
sfd = np.loadtxt('./moon_impact_sfd.txt')
diameter_bins_raw = sfd[:, 0]        # 微陨石尺度 单位km
cumulative_number_raw = sfd[:, 1]	 # 微陨石累积通量 (N>D) km^-2*yr^-1
# max_impacter_diam = diameter_bins_raw[-1] # 默认最大陨石直径
max_impacter_diam = 0.1 # 最大陨石直径
# 微陨石撞击角度分布
impact_angles = np.linspace(0.0, np.pi/2.0, 1000, endpoint=True)
angle_delta = impact_angles[1] - impact_angles[0]
prob = 0.5*np.sin(2.0*impact_angles)
prob[-1] = 0.0
cdf = np.cumsum(prob) # np.cumsum 数据累积求和函数 从概率密度变累积概率密度
angle_cumulative_probability = cdf/np.max(cdf)

# 平均断裂深度和岩石厚度，使用其他地区（如高原）的已知值
# 在模型运行开始时，岩石厚度和未破裂基岩深度均为零。
# 随着运行的进行，这两层厚度随着时间 ∝ t1/2 的增加而增加。
# 在 3.5 Gyr 后产生 5 米厚的岩石层和 8 公里厚的断裂基岩层。
known_avg_fractured_depth = 10000.0
known_avg_age = 4.5e9
model_avg_regolith_thickness = 5.0  # Weber, Encyclopedia of the Solar System, 3rd edition, 2014
now_regolith_thickness = 2.5        # 风化层厚度2.5m
model_avg_age = 4.5e9               # 整个模型的年龄
avg_fractured_depth = (known_avg_fractured_depth / (known_avg_age**(0.5)))*(model_avg_age**(0.5)) # 取决于 √t 来求解与模型区域和年龄相对应的平均厚度

# 连续溅射毯范围 半径的五倍
continuous_ejecta_blanket_factor = 5.0

# 二次坑
secondaries_on = 0
max_secondary_factor = 0.04	# Largest secondary is 4% of primary
v_min_sec = 250.0
min_primary_for_secondaries = (2.0/3.0)*(v_min_sec**2)/g

# 扩散系统
diffusion_on = 1
diffusivity = 0.109

# 粒子追迹
n_particles_per_layer = 10
tracers_on = 1
periodic_particles = 1

# 保存路径
save_dir = ".//Output//"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"文件夹已创建: {save_dir}")
else:
    print(f"文件夹已存在: {save_dir}")
