"""
main.py —— 顶层驱动脚本 / 仿真总流程

职责概述
--------
1) 生成全时段的撞击坑事件清单：
   - 使用 ImpactorPopulation.sample_all_craters() 基于 SFD/角度/速度分布逐步采样；
   - 将每步的坑径与像素坐标落地为一维序列（并保存 CSV 备查）。
2) 构建示踪粒子群：
   - build_tracers_group() 在 (x, y, z) 上规则取样，便于统计层厚/混合深度演化。
3) 逐坑事件更新地形 + 粒子：
   - 对每个时间步中的所有坑，按随机顺序依次调用 Grid.add_crater() 叠加至地形；
   - 每叠加一次坑后，遍历全部粒子，调用 Tracer.tracer_particle_crater() 更新其位置。
4) 汇总与输出：
   - 保存事件时间线 t_line、逐事件的粒子深度分布 partical_depth_all、最终地形 Crater_Map.npy；
   - 最后显示最终地形影像。

数据/单位约定
--------------
- 网格单位：像素；物理长度换算使用 params.resolution（m/px）。
- 坑径/半径：以米（m）计（ImpactorPopulation 内已完成 km↔m 的换算）。
- 时间轴：年（yr）；单步时长为 params.dt。
"""

import params
import numpy as np
from Grid import Grid
from Tracers import build_tracers_group
from ImpactorPopulation import ImpactorPopulation, save_craters
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # 1) 生成全时段的主坑事件清单（直径/中心像素/类型/发生步）
    impPop = ImpactorPopulation()
    d_craters, x_craters, y_craters, index_craters, t_craters = impPop.sample_all_craters()
    # 落地保存一份 CSV（便于事后检查/复现实验）
    save_craters(d_craters, x_craters, y_craters, index_craters, t_craters)

    # 2) 构建示踪粒子群（规则网格化采样）
    tracers = build_tracers_group()

    # 3) 初始化地形网格
    #    - grid_old：当前“已生效”的地形；
    #    - grid_new：每次叠加坑后的临时地形（随后 copy 回 grid_old）
    grid = Grid(params.grid_size, params.resolution, params.diffusivity, params.dt)
    grid_old = grid.setUpGrid()
    grid_new = np.copy(grid_old)

    # 4) 事件时间线与逐事件粒子深度记录
    t_line = []              # 记录每一个“坑事件”的发生时刻（连续时间，单位 yr）
    partical_depth_all = []  # 记录每个事件触发时，所有粒子的深度列表（逐事件入栈）

    # 5) 主时间循环：逐“时间步”（不是逐坑）推进
    for t in range(params.nsteps):
        print("*******************************************************************")
        print("Time step: %d / %d"%(t, params.nsteps))

        # ——— 提取本时间步内所有坑事件（可能为 0 个） ———
        timestep_index = np.where(t_craters == t)
        current_diameters = d_craters[timestep_index]   # 本步坑径（m）
        current_x = x_craters[timestep_index]           # 本步坑中心 x（像素）
        current_y = y_craters[timestep_index]           # 本步坑中心 y（像素）
        current_index = index_craters[timestep_index]   # 类型（此处全 1 表主坑）

        # 为避免空间相关偏序：将本步坑事件随机洗牌后，依次叠加
        index_shuf = list(range(len(current_diameters)))
        np.random.shuffle(index_shuf)
        current_diameters = np.array([current_diameters[j] for j in index_shuf])
        current_x = np.array([current_x[j] for j in index_shuf])
        current_y = np.array([current_y[j] for j in index_shuf])
        current_index = np.array([current_index[j] for j in index_shuf])

        # 本步的连续时间区间 [t*dt, (t+1)*dt)，把该区间平均切给每个坑事件
        t_start = t * params.dt
        imp_dt = params.dt / len(current_diameters) if len(current_diameters) > 0 else params.dt

        # ——— 逐“坑事件”处理：叠加地形 + 更新全部粒子 ———
        for i in range(len(current_diameters)):
            # 事件发生的连续时刻（便于与粒子深度快照对齐）
            imp_t = t_start + imp_dt * i
            t_line.append(imp_t)

            print("\r\tCalculate Crater: %d / %d " % (i, len(current_diameters)), end="")

            # 当前坑参数
            crater_diam = current_diameters[i]       # 坑径（m）
            crater_radius = crater_diam / 2.0        # 坑半径（m）
            crater_index = current_index[i]          # 类型索引（预留）

            # 像素坐标（以数组索引计）
            x_crater_pix = int(current_x[i])
            y_crater_pix = int(current_y[i])

            # 辅助网格（1 矩阵、坐标网格）
            ones_grid = np.ones((params.grid_size, params.grid_size))
            X_grid, Y_grid = np.ogrid[:params.grid_size, :params.grid_size]
            # X_grid 形状 (N, 1): [[0], [1], ..., [N-1]]
            # Y_grid 形状 (1, N): [[0, 1, ..., N-1]]

            # —— 地形叠加：在 grid_old 上添加一个坑（含连续喷出毯），得到 grid_new ——
            grid_new = grid.add_crater(np.copy(grid_old),
                                       x_crater_pix,
                                       y_crater_pix,
                                       crater_diam,
                                       crater_radius,
                                       params.resolution,
                                       params.grid_size,
                                       crater_index,
                                       params.continuous_ejecta_blanket_factor,
                                       X_grid,
                                       Y_grid,
                                       ones_grid)
            # 将叠加后的地形生效为“当前地形”
            np.copyto(grid_old, grid_new)

            # —— 在“叠加了此坑”的新地形上，更新所有粒子的位置 ——
            partical_depth_now = []  # 本坑事件触发瞬间的粒子深度快照（逐粒记录）
            for j in range(len(tracers)):
                particle_position = tracers[j].current_position()
                x_p0 = particle_position[0]
                y_p0 = particle_position[1]
                z_p0 = particle_position[2]

                # 仅处理有效粒子（未越界/未丢失）
                if ~np.isnan(x_p0) and ~np.isnan(y_p0) and ~np.isnan(z_p0):
                    x_p0 = int(x_p0)
                    y_p0 = int(y_p0)
                    z_p0 = z_p0

                    # 当前粒子深度 = 新表面高程(同像素) - 粒子高程
                    d_p0 = grid_old[x_p0, y_p0] - z_p0

                    # 若粒子“飞到地表之上”（数值误差或先前事件导致），先把它放回表面
                    if d_p0 < 0.0:  # 如果颗粒飞腾，让他落回表面
                        z_p0 = grid_old[x_p0, y_p0]
                        d_p0 = grid_old[x_p0, y_p0] - z_p0

                    # 记录本事件快照的粒子深度
                    partical_depth_now.append(d_p0)

                    # 计算粒子到坑心的三维距离分量（dx, dy, dz）
                    dx = (x_p0 - x_crater_pix) * params.resolution
                    dy = (y_p0 - y_crater_pix) * params.resolution
                    if (0 <= x_crater_pix <= (params.grid_size - 1)) and (0 <= y_crater_pix <= (params.grid_size - 1)):
                        dz = z_p0 - grid_old[x_crater_pix, y_crater_pix]
                    else:
                        dz = z_p0
                    R0 = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                    # 仅在“影响范围内”的粒子才进行陨击事件响应更新
                    if R0 <= 5.0 * params.continuous_ejecta_blanket_factor * crater_radius:
                        particle_position_new = tracers[j].tracer_particle_crater(x_p0, y_p0, z_p0, d_p0, dx, dy, dz,
                                                                                  x_crater_pix, y_crater_pix, R0,
                                                                                  crater_radius, grid_old, grid_new)
                        # 就地更新粒子位置
                        tracers[j].update_position(particle_position_new)
            else:
                # 若本步没有任何有效粒子（极少数情况），放一个标记位占位
                partical_depth_now.append(-9999)

            # 事件级别的粒子深度快照入栈
            partical_depth_all.append(partical_depth_now)

        print("\n")

    # 6) 仿真结束：保存时间线、粒子深度序列与最终地形
    t_line = np.array(t_line)
    partical_depth_all = np.array(partical_depth_all)
    np.save(params.save_dir + 't_line.npy', t_line)                # 每个坑事件的发生时刻（yr）
    np.save(params.save_dir + 'partical_depth.npy', partical_depth_all)  # 逐事件的粒子深度快照
    np.save(params.save_dir + 'Crater_Map.npy', grid_new)          # 最终地形（加入所有坑后的高度场）

    # 7) 简单可视化：最终地形影像
    plt.imshow(grid_new)
    plt.show()
