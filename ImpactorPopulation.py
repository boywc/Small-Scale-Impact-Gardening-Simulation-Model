import numpy as np
import params
import csv
from scipy.interpolate import interp1d


def save_craters(d_craters, x_craters, y_craters, index_craters, t_craters):
    csv_file = params.save_dir + 'Craters_Information.csv'
    file = open(csv_file, mode='w', newline='')
    writer = csv.writer(file)
    writer.writerow(["Diameter (m)", "X (pixel)", "Y (pixel)", "Type", "Time step"])
    for i in range(len(d_craters)):
        data = [d_craters[i], x_craters[i], y_craters[i], index_craters[i], t_craters[i]]
        writer.writerow(data)
    file.close()


class ImpactorPopulation:

    def __init__(self):
        pass

    # 基于网格尺度得到的最小撞击坑尺度，获得最小微陨石尺度
    def calc_min_impactor(self, min_crater, v_i, theta_i):
        g = params.g  # 1.623
        delta_i = params.impactor_density  # 2700.0
        rho_t = params.regolith_density  # 1500.0
        R_min = min_crater / 2.0
        nu = 0.4
        # Values for cohesive soils 粘土的值
        k = 1.03
        mu = 0.41
        v_perp = v_i * np.cos(theta_i)
        Y_bar = params.regolith_strength  # 1.0e3   Pa, from Mitchell et al. 1972
        a_min = (R_min / k) * ((Y_bar / (rho_t * (v_perp ** 2))) ** ((2.0 + mu) / 2.0) * (rho_t / delta_i) ** (
                    nu * (2.0 + mu) / mu)) ** (mu / (2.0 + mu))
        # 文章中公式（1）反用，输入撞击坑直径输出微陨石直径，但这里输入的是撞击坑半径，因此输出的也是微陨石的半径
        # return 先将微陨石半径*2转为直径，再/1000 转为km单位
        return (2.0 * a_min) / 1000.0

    # 输入微陨石直径，用粘土的参数计算撞击坑直径
    def pi_group_scale_small(self, d_i, v_i, theta_i, frac_depth):
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

        v_perp = v_i * np.cos(theta_i)

        term1 = g * a_i / (v_perp ** 2)
        term2 = (rho_t / delta_i) ** (2.0 * nu / mu)

        term3 = (Y / (rho_t * v_perp ** 2)) ** ((2.0 + mu) / 2.0)
        term4 = (rho_t / delta_i) ** (nu * (2.0 + mu) / mu)

        crater_radius = k * a_i * ((term1 * term2 + term3 * term4) ** (-mu / (2.0 + mu)))

        return 2.0 * crater_radius

    # 输入微陨石直径，用岩石的参数计算撞击坑直径
    def pi_group_scale_large(self, d_i, v_i, theta_i, frac_depth):
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

        timestep_diams = []
        timestep_x = []
        timestep_y = []
        timestep_primary_index = []
        timestep_time = []
        sec_dist_arr = []

        # 计算当前的风化层厚度和基岩厚度
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
                # 粘土碰撞
                primary_crater_diams = self.pi_group_scale_small(cur_imp_diam * 1000.0, primary_imp_vels * 1000.0, primary_imp_angs, frac_depth)
            else:
                # 岩石碰撞
                primary_crater_diams = self.pi_group_scale_large(cur_imp_diam * 1000.0, primary_imp_vels * 1000.0, primary_imp_angs, frac_depth)

            # 滤除生成的小于分辨率的撞击坑
            min_crater_mask = np.where(primary_crater_diams >= min_crater)
            primary_crater_diams = primary_crater_diams[min_crater_mask]

            # 如果当前有撞击坑生成
            if len(primary_crater_diams) > 0:

                # 随机生成撞击坑距离中心的位置与角度，计算x, y坐标
                primary_crater_dists = max_grid_dist[i] * np.sqrt(np.random.uniform(size=len(primary_crater_diams)))
                primary_crater_phis = np.random.uniform(size=len(primary_crater_diams)) * 2.0 * np.pi
                primary_crater_x = primary_crater_dists * np.cos(primary_crater_phis)
                primary_crater_y = primary_crater_dists * np.sin(primary_crater_phis)
                primary_crater_x_pix = np.array(primary_crater_x / resolution + grid_size / 2.0).astype('int')
                primary_crater_y_pix = np.array(-primary_crater_y / resolution + grid_size / 2.0).astype('int')

                # 存储撞击坑信息
                timestep_diams.append(primary_crater_diams)   # 存入撞击坑直径
                timestep_x.append(primary_crater_x_pix)       # 存入撞击坑像素位置
                timestep_y.append(primary_crater_y_pix)       # 存入撞击坑像素位置
                timestep_primary_index.append(np.ones(len(primary_crater_diams)))   # 存入全1列表




        if len(timestep_diams) > 0:

            timestep_diams = np.array(np.concatenate(timestep_diams))
            timestep_x = np.array(np.concatenate(timestep_x))
            timestep_y = np.array(np.concatenate(timestep_y))
            timestep_primary_index = np.array(np.concatenate(timestep_primary_index))
            timestep_time = t * np.ones(len(timestep_diams))

        return timestep_diams, timestep_x, timestep_y, timestep_primary_index, timestep_time

    # 生成撞击调用坑方法
    def sample_all_craters(self):

            nsteps = params.nsteps  # 仿真运行的时间步数
            grid_width = params.grid_width  # 网格像素尺度
            X_grid, Y_grid = np.ogrid[:params.grid_size, :params.grid_size]  # 生成(n, 1)和(1, n)两个x, y数组

            d_craters = []
            x_craters = []
            y_craters = []
            index_craters = []
            t_craters = []

            for t in range(nsteps):
                t_time = t * params.dt  # 当前模拟时间
                d_min = self.calc_min_impactor(params.min_crater, params.velocity_max*1000.0, 0.0)  # 最小微陨石直径 单位km

                # 得到微陨石直径表 （单位 km）
                diam_bins = [d_min]  # 单位km
                while diam_bins[-1] < params.max_impacter_diam:
                    # 生成微陨石直径表，区间范围：从最小陨石到数据库中统计最大陨石，等比数列间隔：√2
                    diam_bins.append(np.sqrt(2.0)*diam_bins[-1])
                diam_bins = np.array(diam_bins)
                avg_imp_diam = diam_bins*(2.0**(1.0/4.0))   # 平均微陨石直径 单位km
                # 内插法得到陨石尺度分区，从 SFD 中获得每个分区的累计数
                f = interp1d(np.log10(params.diameter_bins_raw), np.log10(params.cumulative_number_raw), fill_value='extrapolate')
                cum_num = 10 ** f(np.log10(diam_bins))
                num_inc = cum_num[:-1] - cum_num[1:]  # 累积密度转为密度 /m2/Myr 注：与求导不同，这个是当前的减去下一个
                avg_imp_diam = avg_imp_diam[0: -1]  # 对应撞击密度的微陨石直径表 单位km
                diam_bins = diam_bins[0: -1]

                # 基于陨石，粗略计算撞击坑直径表 （单位 m）
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

                # 接收撞击区域
                # 网格区域 + 在网格外撞但也可击影响网格的区域         网格的对角线/2 + 半径五倍皆可覆盖
                max_grid_dist = (np.sqrt(grid_width**2/2.0) + params.continuous_ejecta_blanket_factor*max_crater_radius)
                max_grid_area = np.pi*max_grid_dist**2  # 最大撞击坑覆盖的面积 m^2

                # 计算陨石个数表
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

            d_craters = np.array(np.concatenate(d_craters))
            x_craters = np.array(np.concatenate(x_craters))
            y_craters = np.array(np.concatenate(y_craters))
            index_craters = np.array(np.concatenate(index_craters))
            t_craters = np.array(np.concatenate(t_craters)).astype('int')

            return d_craters, x_craters, y_craters, index_craters, t_craters


