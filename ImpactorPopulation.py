"""
ImpactorPopulation：微陨石（撞击体）群的采样与撞击坑（Crater）事件生成

本模块职责
----------
1) 基于给定的速度分布、入射角分布、尺度-频率分布（SFD），在每个时间步对落入/影响网格的
   微陨石进行随机采样（泊松过程），并据此计算对应的撞击坑直径、在网格上的投影坐标等；
2) 提供两套缩尺律（π分组/相似律）以在“粘土参数（一致于风化层）”与“岩石参数（基岩）”
   条件下将微陨石直径 -> 撞击坑直径；
3) 依据网格与连续喷出毯范围，估算“有效影响区域”，从而将面积通量换算到每步期望事件数；
4) 输出本步/全程的撞击坑清单，可用于后续地形演化（坑叠加、地形扩散等）。

外部依赖
--------
- `params.py`：提供物理常量、SFD/速度分布、网格与时间设置等
- `numpy / scipy.interpolate.interp1d`

单位约定（除非特别说明）
----------------------
- 长度：米 m（外部 SFD/直径表以 km 给出时，会在使用处显式换算）
- 时间：年 yr
- 速度：m/s（外部速度分布若以 km/s 给出，使用前需 ×1000）
- 角度：弧度（入射角 θ，θ=0 表示水平入射；θ=π/2 表示垂直入射）

"""

import numpy as np
import params
import csv
from scipy.interpolate import interp1d


def save_craters(d_craters, x_craters, y_craters, index_craters, t_craters):
    """
    将生成的撞击坑清单保存为 CSV（直径、像素坐标、类型、时间步）。

    Args:
        d_craters (ndarray or list): 撞击坑直径（m）。
        x_craters (ndarray or list): 撞击坑中心的 x 坐标（像素）。
        y_craters (ndarray or list): 撞击坑中心的 y 坐标（像素）。
        index_craters (ndarray or list): 类型标记（此处 1 表示主坑；若扩展可支持二次坑等）。
        t_craters (ndarray or list): 发生的时间步索引（int）。
    """
    csv_file = params.save_dir + 'Craters_Information.csv'
    file = open(csv_file, mode='w', newline='')
    writer = csv.writer(file)
    writer.writerow(["Diameter (m)", "X (pixel)", "Y (pixel)", "Type", "Time step"])
    for i in range(len(d_craters)):
        data = [d_craters[i], x_craters[i], y_craters[i], index_craters[i], t_craters[i]]
        writer.writerow(data)
    file.close()


class ImpactorPopulation:
    """
    微陨石群体采样器。

    提供：
    - 最小撞击体估算（由网格可分辨的最小坑反推最小微陨石）
    - π分组缩尺律（粘土/岩石两套参数）
    - 速度/角度的随机采样（按外部经验分布）
    - 在单个时间步内生成具体的“坑事件”（直径、坐标、类型、时间）
    - 迭代全部时间步，汇总全程撞击坑清单
    """

    def __init__(self):
        """当前类不需要在构造阶段传入参数（所有配置从 params 读取）。"""
        pass

    # 基于网格尺度得到的最小撞击坑尺度，获得最小微陨石尺度
    def calc_min_impactor(self, min_crater, v_i, theta_i):
        """
        由“可分辨的最小坑直径”反推“最小微陨石直径”（粘性土参数组）。

        采用 π 分组缩尺律的“反问题”思路：给定最小坑半径 R_min 与入射条件，求微陨石半径 a_min，
        再返回其直径（并换算为 km）。

        Args:
            min_crater (float): 最小坑直径（m）。
            v_i (float): 入射速度（m/s）。
            theta_i (float): 入射角（弧度），与地平线夹角；用 v_perp = v_i * cos(theta)。

        Returns:
            float: 最小微陨石直径（km）。

        Notes:
            - 目标物性采用风化层（粘土）参数：k=1.03, mu=0.41, nu=0.4；
            - 回传单位为 km（后续 SFD 以 km 计较为方便）。
        """
        g = params.g  # 1.623
        delta_i = params.impactor_density  # 2700.0
        rho_t = params.regolith_density  # 1500.0
        R_min = min_crater / 2.0
        nu = 0.4
        # Values for cohesive soils 粘土的值
        k = 1.03
        mu = 0.41
        v_perp = v_i * np.cos(theta_i)  # 垂直分量决定有效动压
        Y_bar = params.regolith_strength  # 1.0e3   Pa, from Mitchell et al. 1972

        # 反解 a_min（单位：m），随后 ×2 得直径，再 /1000 转 km
        a_min = (R_min / k) * ((Y_bar / (rho_t * (v_perp ** 2))) ** ((2.0 + mu) / 2.0) * (rho_t / delta_i) ** (
                    nu * (2.0 + mu) / mu)) ** (mu / (2.0 + mu))

        # 文章中公式（1）反用，输入撞击坑直径输出微陨石直径，但这里输入的是撞击坑半径，因此输出的也是微陨石的半径
        # return 先将微陨石半径*2转为直径，再/1000 转为km单位
        return (2.0 * a_min) / 1000.0

    # 输入微陨石直径，用粘土的参数计算撞击坑直径
    def pi_group_scale_small(self, d_i, v_i, theta_i, frac_depth):
        """
        π 分组缩尺律（风化层/粘土参数）：微陨石直径 -> 撞击坑直径。

        Args:
            d_i (float or ndarray): 微陨石直径（m）。
            v_i (float or ndarray): 入射速度（m/s）。
            theta_i (float or ndarray): 入射角（弧度）。
            frac_depth (float): 断裂层/风化层有效厚度（m），用于插值目标强度/密度。

        Returns:
            float or ndarray: 撞击坑直径（m）。

        Notes:
            - k=1.03, mu=0.41（粘性土）；nu=0.4；
            - 根据 a_i 与 frac_depth 的大小关系在线性插值目标强度/密度（风化层到基岩过渡）。
        """
        # d_i = impactor diameter in m
        # v_i = impactor velocity in m/s
        # target = target body (moon, ceres, mercury)
        # delta_i = impactor density, 2.7 g/cm^3 from Marchi et al. 2009
        # theta_i = impact angle

        # Returns final crater diameter in meters*****

        g = params.g
        delta_i = params.impactor_density
        rho_surf = params.regolith_density
        rho_depth = params.bedrock_density
        Y_surf = params.regolith_strength
        Y_depth = params.bedrock_strength

        a_i = d_i / 2.0
        # 目标强度/密度：随深度在风化层与基岩之间插值
        if frac_depth == 0.0:
            Y = Y_depth
            rho_t = rho_depth
        elif 10.0 * a_i <= frac_depth:
            Y = ((Y_depth - Y_surf) / frac_depth) * (5.0 * a_i) + 0.0
            rho_t = ((rho_depth - rho_surf) / frac_depth) * (5.0 * a_i) + rho_surf
        elif 10.0 * a_i > frac_depth:
            Y = ((Y_depth + Y_surf) / 2.0) * (frac_depth / (10.0 * a_i)) + Y_depth * (10.0 * a_i - frac_depth) / (
                        10.0 * a_i)
            rho_t = ((rho_depth + rho_surf) / 2.0) * (frac_depth / (10.0 * a_i)) + rho_depth * (
                        10.0 * a_i - frac_depth) / (10.0 * a_i)

        # Values for cohesive soils
        k = 1.03
        mu = 0.41

        nu = 0.4

        v_perp = v_i * np.cos(theta_i)  # 垂直分量

        # π组分：重力项与强度项（两者并联合成）
        term1 = g * a_i / (v_perp ** 2)
        term2 = (rho_t / delta_i) ** (2.0 * nu / mu)

        term3 = (Y / (rho_t * v_perp ** 2)) ** ((2.0 + mu) / 2.0)
        term4 = (rho_t / delta_i) ** (nu * (2.0 + mu) / mu)

        crater_radius = k * a_i * ((term1 * term2 + term3 * term4) ** (-mu / (2.0 + mu)))

        return 2.0 * crater_radius

    # 输入微陨石直径，用岩石的参数计算撞击坑直径
    def pi_group_scale_large(self, d_i, v_i, theta_i, frac_depth):
        """
        π 分组缩尺律（岩石/基岩参数）：微陨石直径 -> 撞击坑直径。

        与 `pi_group_scale_small` 类似，但参数采用岩石值：k=0.93, mu=0.55。

        Args:
            d_i (float or ndarray): 微陨石直径（m）。
            v_i (float or ndarray): 入射速度（m/s）。
            theta_i (float or ndarray): 入射角（弧度）。
            frac_depth (float): 断裂层/风化层有效厚度（m）。

        Returns:
            float or ndarray: 撞击坑直径（m）。
        """
        # d_i = impactor diameter in m
        # v_i = impactor velocity in m/s
        # target = target body (moon, ceres, mercury)
        # delta_i = impactor density, 2.7 g/cm^3 from Marchi et al. 2009
        # theta_i = impact angle

        # Returns final crater diameter in meters*****

        g = params.g
        delta_i = params.impactor_density
        rho_surf = params.regolith_density
        rho_depth = params.bedrock_density
        Y_surf = params.regolith_strength
        Y_depth = params.bedrock_strength

        a_i = d_i / 2.0

        # 目标强度/密度插值（与 small 版本一致）
        if frac_depth == 0.0:
            Y = Y_depth
            rho_t = rho_depth
        elif 10.0 * a_i <= frac_depth:
            Y = ((Y_depth - Y_surf) / frac_depth) * (5.0 * a_i) + 0.0
            rho_t = ((rho_depth - rho_surf) / frac_depth) * (5.0 * a_i) + rho_surf
        elif 10.0 * a_i > frac_depth:
            Y = ((Y_depth + Y_surf) / 2.0) * (frac_depth / (10.0 * a_i)) + Y_depth * (10.0 * a_i - frac_depth) / (
                        10.0 * a_i)
            rho_t = ((rho_depth + rho_surf) / 2.0) * (frac_depth / (10.0 * a_i)) + rho_depth * (
                        10.0 * a_i - frac_depth) / (10.0 * a_i)

        # Values for rock
        k = 0.93
        mu = 0.55

        nu = 0.4

        v_perp = v_i * np.cos(theta_i)

        term1 = g * a_i / (v_perp ** 2)
        term2 = (rho_t / delta_i) ** (2.0 * nu / mu)

        term3 = (Y / (rho_t * v_perp ** 2)) ** ((2.0 + mu) / 2.0)
        term4 = (rho_t / delta_i) ** (nu * (2.0 + mu) / mu)

        crater_radius = k * a_i * ((term1 * term2 + term3 * term4) ** (-mu / (2.0 + mu)))

        return 2.0 * crater_radius

    # 随机生成"速度"
    def sample_impact_velocities(self, n_samples=1):
        """
        从经验速度分布（CDF）采样入射速度（km/s），并作微扰以避免“离散格点”伪影。

        过程：
            1) 在 [0,1] 上均匀采样 R；
            2) 从 `cum_prob_vels` 中找到最接近 R 的速度格点；
            3) 在该速度区间内再加一个均匀微扰 ∈ [-Δv, +Δv] 以打散格点效应；
            4) 返回采样结果（仍为 km/s；若用于缩尺律，请在调用处×1000 转 m/s）。

        Args:
            n_samples (int): 采样数量。

        Returns:
            ndarray: 采样速度（km/s）。
        """
        vels = params.velocities
        cum_prob_vels = params.velocity_cumulative_probability
        d_vel = params.velocity_delta
        # 双重随机，先随机选取一个速度区间，再在这个区间随机选取一个速度
        R_prob = np.random.uniform(0.0, 1.0, n_samples)
        add_rand = np.random.uniform(-d_vel, d_vel, n_samples)
        gen_vels = [vels[np.argmin((cum_prob_vels - r)**2)] for r in R_prob]
        gen_vels = gen_vels + add_rand
        return np.array(gen_vels)

    # 随机生成"角度"
    def sample_impact_angles(self, n_samples=1):
        """
        从入射角分布（CDF）采样角度 θ（弧度），并作微扰以减少格点集中。

        分布来源：各向同性入射 -> p(θ) = ½ sin(2θ)，θ ∈ [0, π/2]。

        Args:
            n_samples (int): 采样数量。

        Returns:
            ndarray: 采样角度（弧度）。
        """
        angs = params.impact_angles
        cum_prob_angs = params.angle_cumulative_probability
        d_ang = params.angle_delta
        # 双重随机，先随机选取一个速度区间，再在这个区间随机选取一个速度
        R_prob = np.random.uniform(0.0, 1.0, n_samples)
        add_rand = np.random.uniform(-d_ang, d_ang, n_samples)
        gen_angs = [angs[np.argmin((cum_prob_angs - r)**2)] for r in R_prob]
        gen_angs = gen_angs + add_rand
        return np.array(gen_angs)

    # 通过预估参数，基于泊松分布，生成具体的撞击坑
    def sample_timestep_craters(self, t, avg_imp_diam, primary_lams, max_grid_dist, avg_crater_diam, num_inc, dt,
                                min_crater, resolution, grid_size, grid_width, min_primary_for_secondaries,
                                secondaries_on, r_body, continuous_ejecta_blanket_factor, X_grid, Y_grid):
        """
        在单个时间步内：按泊松过程生成“主坑”并投影到网格坐标。

        核心步骤：
            1) 对每个微陨石直径分段 i，从 Poisson(λ_i) 采样主坑数量；
            2) 为每个事件采样速度/角度，使用缩尺律得到坑径（m），滤掉 < min_crater 的小坑；
            3) 在“最大影响半径 max_grid_dist[i]”内均匀取平面点（极坐标均匀：r ~ sqrt(U)）；
            4) 变换到像素坐标（以网格中心为原点），并记录类型与时间步。

        Args:
            t (int): 当前时间步索引。
            avg_imp_diam (ndarray): 分段平均微陨石直径（km）。
            primary_lams (ndarray): 各分段主坑的期望个数（Poisson 参数 λ_i）。
            max_grid_dist (ndarray): 对应分段的最大影响半径（m），含喷出毯。
            avg_crater_diam (ndarray): 估计的平均坑径（m），此处仅作为参考/调试输入。
            num_inc (ndarray): 分段密度（单位 m^-2·Myr^-1），来自累计通量差分。
            dt (float): 单步时长（yr）。
            min_crater (float): 最小可分辨坑径（m）。
            resolution (float): 网格分辨率（m/px）。
            grid_size (int): 网格边长（px）。
            grid_width (float): 物理域宽（m）。
            min_primary_for_secondaries (float): 二次坑阈值相关参数（本函数中未使用，预留）。
            secondaries_on (int): 是否启用二次坑（本函数中未使用，预留）。
            r_body (float): 天体半径（m）（本函数中未使用，预留）。
            continuous_ejecta_blanket_factor (float): 连续喷出毯半径系数（本函数中未直接使用）。
            X_grid, Y_grid (ndarray): ogrid 生成的网格坐标辅助（此处未直接使用，预留）。

        Returns:
            tuple:
                timestep_diams (ndarray): 本步所有主坑直径（m）。
                timestep_x (ndarray): 像素 x 坐标（int）。
                timestep_y (ndarray): 像素 y 坐标（int）。
                timestep_primary_index (ndarray): 类型标记（全 1，表示主坑）。
                timestep_time (ndarray): 时间步索引（同长，值为 t）。

        Notes:
            - 采用极坐标均匀采样：r = R_max * sqrt(U)，phi = 2πU，以保证面密度均匀。
            - 速度/角度采样返回单位分别为 km/s 与弧度；缩尺律调用时已将 km/s→m/s。
        """
        timestep_diams = []
        timestep_x = []
        timestep_y = []
        timestep_primary_index = []
        timestep_time = []
        sec_dist_arr = []  # 预留：若后续引入二次坑距离统计，这里可直接复用

        # 计算当前的风化层厚度和基岩厚度（此处取固定：风化层厚度与最大断裂深度）
        regolith_thickness = params.now_regolith_thickness  # 风化层厚度使用CE6地区风化层 2.5m
        frac_depth = params.known_avg_fractured_depth       # 仿真最后一段时期的撞击情况，断裂深度设置为最深

        for i in range(len(avg_imp_diam)):  # 遍历撞击器直径区间
            cur_imp_diam = avg_imp_diam[i]

            # 通过陨石直径（km），基于泊松分布，生成陨石数量、撞击速度、角度
            num_cur_grid_imps = np.random.poisson(lam=primary_lams[i])           # 按泊松分布得到撞击器数量 (随机数)
            primary_imp_vels = self.sample_impact_velocities(num_cur_grid_imps)  # 生成速度列表 km/s
            primary_imp_angs = self.sample_impact_angles(num_cur_grid_imps)      # 生成撞击角度列表

            # 通过陨石计算撞击坑尺度 （m）
            if cur_imp_diam * 1000.0 <= (1.0 / 20.0) * frac_depth:               # 如果陨石（km单位）直径 <= 基岩的1/20
                # 粘土碰撞（速度单位转 m/s）
                primary_crater_diams = self.pi_group_scale_small(cur_imp_diam * 1000.0, primary_imp_vels * 1000.0, primary_imp_angs, frac_depth)
            else:
                # 岩石碰撞
                primary_crater_diams = self.pi_group_scale_large(cur_imp_diam * 1000.0, primary_imp_vels * 1000.0, primary_imp_angs, frac_depth)

            # 滤除生成的小于分辨率的撞击坑
            min_crater_mask = np.where(primary_crater_diams >= min_crater)
            primary_crater_diams = primary_crater_diams[min_crater_mask]

            # 如果当前有撞击坑生成
            if len(primary_crater_diams) > 0:

                # 在最大影响半径内均匀采样平面点：r ~ sqrt(U)，phi ~ U
                primary_crater_dists = max_grid_dist[i] * np.sqrt(np.random.uniform(size=len(primary_crater_diams)))
                primary_crater_phis = np.random.uniform(size=len(primary_crater_diams)) * 2.0 * np.pi
                primary_crater_x = primary_crater_dists * np.cos(primary_crater_phis)
                primary_crater_y = primary_crater_dists * np.sin(primary_crater_phis)

                # 变换为像素坐标（以网格中心为原点，y 轴取屏幕坐标朝下为正）
                primary_crater_x_pix = np.array(primary_crater_x / resolution + grid_size / 2.0).astype('int')
                primary_crater_y_pix = np.array(-primary_crater_y / resolution + grid_size / 2.0).astype('int')

                # 存储撞击坑信息
                timestep_diams.append(primary_crater_diams)   # 存入撞击坑直径
                timestep_x.append(primary_crater_x_pix)       # 存入撞击坑像素位置
                timestep_y.append(primary_crater_y_pix)       # 存入撞击坑像素位置
                timestep_primary_index.append(np.ones(len(primary_crater_diams)))   # 存入全1列表

        # 将分段列表拼接成一维数组，并配齐时间步索引
        if len(timestep_diams) > 0:

            timestep_diams = np.array(np.concatenate(timestep_diams))
            timestep_x = np.array(np.concatenate(timestep_x))
            timestep_y = np.array(np.concatenate(timestep_y))
            timestep_primary_index = np.array(np.concatenate(timestep_primary_index))
            timestep_time = t * np.ones(len(timestep_diams))

        return timestep_diams, timestep_x, timestep_y, timestep_primary_index, timestep_time

    # 生成撞击调用坑方法
    def sample_all_craters(self):
        """
        在整个模拟时段内迭代生成所有时间步的撞击坑清单。

        步骤概要：
            1) 读取全局参数（nsteps、网格尺度、分布/常量等）；
            2) 为每个时间步：
               2.1) 根据最小可分辨坑径 -> 反推最小微陨石直径（km）；
               2.2) 构造微陨石直径分段（等比 √2 间隔），与 SFD 累计通量插值 -> 分段密度；
               2.3) 以分段平均直径估算“代表坑径”（max/avg，供影响半径估算与调试使用）；
               2.4) 估算最大影响半径与面积（含喷出毯范围因子）；
               2.5) 将面积通量与时间步长换算为各分段的 Poisson λ；
               2.6) 调用 `sample_timestep_craters` 生成本步坑事件；
            3) 拼接所有时间步结果并返回。

        Returns:
            tuple:
                d_craters (ndarray): 全程所有主坑直径（m）。
                x_craters (ndarray): 像素 x 坐标（int）。
                y_craters (ndarray): 像素 y 坐标（int）。
                index_craters (ndarray): 类型标记（全 1）。
                t_craters (ndarray): 发生时间步（int）。
        """
        nsteps = params.nsteps  # 仿真运行的时间步数
        grid_width = params.grid_width  # 网格像素尺度
        X_grid, Y_grid = np.ogrid[:params.grid_size, :params.grid_size]  # 生成(n, 1)和(1, n)两个x, y数组

        d_craters = []
        x_craters = []
        y_craters = []
        index_craters = []
        t_craters = []

        for t in range(nsteps):
            t_time = t * params.dt  # 当前模拟时间（yr）（此处仅作为参考，未直接写入结果）
            # 由最小坑径反推最小微陨石直径（km）
            d_min = self.calc_min_impactor(params.min_crater, params.velocity_max*1000.0, 0.0)  # 最小微陨石直径 单位km

            # 得到微陨石直径表 （单位 km），等比序列，公比 √2
            diam_bins = [d_min]  # 单位km
            while diam_bins[-1] < params.max_impacter_diam:
                # 生成微陨石直径表，区间范围：从最小陨石到数据库中统计最大陨石，等比数列间隔：√2
                diam_bins.append(np.sqrt(2.0)*diam_bins[-1])
            diam_bins = np.array(diam_bins)
            avg_imp_diam = diam_bins*(2.0**(1.0/4.0))   # 平均微陨石直径 单位km（几何中点）

            # 内插 SFD 的累计通量到这些分段边界，并由差分得到“分段密度”
            f = interp1d(np.log10(params.diameter_bins_raw), np.log10(params.cumulative_number_raw), fill_value='extrapolate')
            cum_num = 10 ** f(np.log10(diam_bins))
            num_inc = cum_num[:-1] - cum_num[1:]  # 累积密度转为密度 /m2/Myr 注：非求导，是相邻分段差
            avg_imp_diam = avg_imp_diam[0: -1]  # 与 num_inc 对齐
            diam_bins = diam_bins[0: -1]

            # 以平均微陨石直径估算代表坑径（供“影响半径”与调试参考）
            max_crater_diam = np.zeros(len(diam_bins))
            avg_crater_diam = np.zeros(len(diam_bins))
            for i in range(len(diam_bins)):
                if avg_imp_diam[i] * 1000.0 <= (1.0 / 20.0) * params.avg_fractured_depth:
                    max_crater_diam[i] = self.pi_group_scale_small(avg_imp_diam[i]*1000.0, params.velocity_max*1000.0, 0.0, params.avg_fractured_depth)
                    avg_crater_diam[i] = self.pi_group_scale_small(avg_imp_diam[i]*1000.0, params.velocity_average*1000.0, np.deg2rad(45.0), params.avg_fractured_depth)
                else:
                    max_crater_diam[i] = self.pi_group_scale_large(avg_imp_diam[i] * 1000.0, params.velocity_max * 1000.0, 0.0, params.avg_fractured_depth)
                    avg_crater_diam[i] = self.pi_group_scale_large(avg_imp_diam[i] * 1000.0, params.velocity_average * 1000.0, np.deg2rad(45.0), params.avg_fractured_depth)
            max_crater_radius = max_crater_diam / 2.0

            # 接收撞击区域（网格本身 + 喷出毯可能影响到的外围）
            # 取网格对角线的一半加上 5×半径（典型喷出毯范围），近似覆盖所有可能影响
            max_grid_dist = (np.sqrt(grid_width**2/2.0) + params.continuous_ejecta_blanket_factor*max_crater_radius)
            max_grid_area = np.pi*max_grid_dist**2  # 最大撞击坑覆盖的面积 m^2

            # 将面积通量（/m^2/Myr）换算为单步期望事件数（Poisson λ）
            primary_lams = num_inc * (max_grid_area/(1.0e6)) * params.dt  # num_inc单位 /m2/Myr 直接*m^2面积 用y/1e6年龄

            # 随机化过程，利用上面的估计参数，基于泊松分布，生成具体的撞击坑，位置，形态等
            return_inf = self.sample_timestep_craters(
                t,                    # 当前模拟时间步数 单位1
                avg_imp_diam,         # 微陨石直径表 单位km
                primary_lams,         # 撞击坑个数表 单位1
                max_grid_dist,        # 接收撞击区域直径表 单位m
                avg_crater_diam,      # 撞击坑平均直径表 单位m
                num_inc,              # 概率密度表 单位/m2/Myr
                params.dt,
                params.min_crater,
                params.resolution,
                params.grid_size,
                params.grid_width,
                params.min_primary_for_secondaries,
                params.secondaries_on,
                params.radius_body,
                params.continuous_ejecta_blanket_factor,
                X_grid,
                Y_grid)
            timestep_diams, timestep_x, timestep_y, timestep_primary_index, timestep_time = return_inf

            d_craters.append(timestep_diams)
            x_craters.append(timestep_x)
            y_craters.append(timestep_y)
            index_craters.append(timestep_primary_index)
            t_craters.append(timestep_time)

        # 拼接所有时间步的结果为一维数组
        d_craters = np.array(np.concatenate(d_craters))
        x_craters = np.array(np.concatenate(x_craters))
        y_craters = np.array(np.concatenate(y_craters))
        index_craters = np.array(np.concatenate(index_craters))
        t_craters = np.array(np.concatenate(t_craters)).astype('int')

        return d_craters, x_craters, y_craters, index_craters, t_craters
