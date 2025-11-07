# Small-Scale Impact Gardening Simulation (CE-6)

**月球表层小尺度三维数值模拟（Chang’E-6 登陆区场景）**
本仓库提供用于研究“微陨石长期轰击引起的风化层搅动与混合”的三维模拟代码。

该实现基于 O’Brien 等人的开源模型进行定制化修改，以适配 CE-6 远端登陆区的小尺度地形格局与入流条件；并在此基础上增加了示踪粒子统计、事件时间线与 3D 粒子曝露龄评估等功能。
实现包含：
* 基于速度/入射角/SFD（尺度-频率分布）的**泊松事件采样**；
* DEM 网格上的**陨击坑+连续喷出毯**叠加与**显/隐式扩散**；
* **示踪粒子**在扩散与陨击事件下的运动学更新；
* 3D 球面体素模型用于**粒子曝露龄**（含自遮蔽的 1/4 可视化修正）。
---

## 主要特性

* **微陨石事件生成**：按速度分布、入射角分布与尺度-频率分布（SFD）在每个时间步泊松采样，得到主坑事件时间线与空间分布。
* **地形过程耦合**：网格显/隐式扩散 + 陨击坑/喷出毯解析剖面叠加，含体量守恒尺度因子。
* **示踪粒子统计**：支持表面弹射、地下喷射、连续喷出掩埋与钻孔式下沉等情景，逐事件输出“粒子深度快照”。
* **结果可视化**：提供 DEM 演化、粒子轨迹、样品/粒子深度窗口内来源统计等绘图脚本。
* **3D 粒子曝露龄**：`grain_3D.py` 给出 3D 粒子曝光模拟与可视化，并说明**绘图阶段将曝露龄÷4**的经验修正（遮蔽效应）来源与做法。

---

## 目录结构

```plaintext
.
├─ main.py                 # 主程序：驱动事件采样 → 地形演化 → 粒子更新
├─ params.py               # 仿真全局参数（单位/依赖/默认值）
├─ Tracers.py              # 构建示踪粒子层与运动跟踪（扩散/陨击响应）
├─ Grid.py                 # 网格与地形：扩散、Crater+喷出毯叠加、守恒缩放
├─ ImpactorPopulation.py   # 微陨石群体采样与泊松事件生成（速度/角度/SFD）
├─ grain_3D.py             # 3D 粒子曝露龄演示与“1/4 修正”来源说明
├─ display/                # 可视化脚本
│  ├─ draw_impact_dem.py
│  ├─ draw_track.py
│  └─ draw_sample_grains.py
└─ Output/                 # 仿真输出目录（自动创建）
```
---

## 环境与依赖

* Python 3.8+
* `numpy`, `scipy`, `matplotlib`

```bash
pip install numpy scipy matplotlib
```

---

## 模块概览

* **`ImpactorPopulation.py`**：速度/角度/SFD 抽样 → 分段面积通量 → 泊松采样 → 本步主坑列表
* **`Grid.py`**：ADI/显式扩散、Crater+喷出毯叠加、守恒尺度因子计算
* **`Tracers.py`**：粒子更新（扩散/表面弹射/地下喷射/掩埋/钻孔）与历史轨迹记录
* **`main.py`**：时间步循环；按事件顺序叠加地形并更新全部粒子；写出事件时间线与 DEM

---

## 快速上手

1. 调整 `params.py` 中的分辨率/时间步长/物性参数等；
2. 执行主程序：

```bash
python main.py
```

3. 查看输出（默认写入 `Output/`）：

| 文件名                       | 含义说明                     |
| ------------------------- | ------------------------ |
| `Craters_Information.csv` | 每个陨击坑事件的直径、类型、像素坐标、发生时间步 |
| `t_line.npy`              | 逐“坑事件”的物理时间戳（year）       |
| `partical_depth.npy`      | 事件触发瞬间的“全体粒子深度快照”序列（逐事件） |
| `Crater_Map.npy`          | 仿真结束时的 DEM（叠加所有事件后的高度场）  |

---

## API Reference · 函数速查（按模块）

> 说明：以下签名与行为以源码为准；参数单位除非特别说明，长度用米 m（像素换算由 `params.resolution` 完成）、时间 year、速度 m/s、角度弧度。
> 单元格内多参数时以分号分隔；返回值列给出结构与单位。

---

### 1. main.py API 参考手册

| 方法名 / 入口               | 功能描述                          | 参数说明                                                                                                                | 返回值 / 输出                                                                                    |
| ---------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `python main.py`（脚本入口） | 运行整段仿真流程（采样 → 叠加 → 粒子更新 → 输出） | 读取 `params.py`；内部调用 `ImpactorPopulation.sample_all_craters()`、`Grid.add_crater()`、`Tracer.tracer_particle_crater()` | 写出 `Craters_Information.csv`、`t_line.npy`、`partical_depth.npy`、`Crater_Map.npy`；并弹出最终 DEM 图 |

> 本文件无对外函数，作为 CLI 入口脚本存在。

---

### 2. params.py API 参考手册（关键配置项）

| 名称（示例）                                   | 功能描述               | 典型取值 / 单位                              | 备注                               |
| ---------------------------------------- | ------------------ | -------------------------------------- | -------------------------------- |
| `resolution`                             | 网格分辨率（m/px）        | `0.01` m/px                            | 与 `grid_width` 一起决定 `grid_size`  |
| `dt`, `model_time`, `nsteps`             | 时间步长 / 总时长 / 步数    | `1e6` yr；`2e7` yr；`model_time/dt`      | 显式扩散需满足 CFL；隐式/Crank–Nicolson 较稳 |
| `grid_width`, `grid_size`                | 物理域宽（m）/ 网格边长（px）  | `2.0` m；`grid_width/resolution`        | 方形网格                             |
| `diffusivity`, `diffusion_on`            | 扩散系数 / 隐式开关        | `0.109`；`1`                            | 隐式分步解三对角；显式为 FTCS                |
| `impactor_density` 等密度/强度                | 撞击体/风化层/基岩密度与强度    | `2700/1500/3148.57 kg/m³`；`1e3/2e7 Pa` | 用于 π 分组缩尺律                       |
| `velocities` & CDF、`impact_angles` & CDF | 入射速度与角度的离散分布及其累计分布 | 速度按 km/s 表；角度 `0~π/2`                  | 速度在缩尺计算处转 m/s                    |
| `continuous_ejecta_blanket_factor`       | 连续喷出毯半径系数（×坑半径）    | `5.0`                                  | 影响“最大作用半径”估算                     |
| `n_particles_per_layer`                  | 每层示踪粒子密度（行×列）      | `10`                                   | 垂向默认 100 层                       |
| `save_dir`                               | 输出目录               | `./Output/`                            | 运行时自动创建                          |

---

### 3. Grid.py API 参考手册

| 方法名                                                                                                                                                  | 功能描述                                                | 参数说明                                                                                                                                                                                                       | 返回值                                |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `class Grid(grid_size, resolution, diffusivity, dt)`                                                                                                 | 网格容器与数值算子：初始化隐式扩散三对角带状矩阵与系数 `R`。                    | `grid_size`: 网格边长 px；`resolution`: m/px；`diffusivity`: D；`dt`: Δt yr                                                                                                                                       | `Grid` 实例                          |
| `setUpGrid()`                                                                                                                                        | 创建初始 DEM（零场）。                                       | 无                                                                                                                                                                                                          | `np.ndarray[grid_size, grid_size]` |
| `explicit_diffusion2D(u0)`                                                                                                                           | 显式 FTCS 扩散一步，边界镜像处理。                                | `u0`: 当前 DEM                                                                                                                                                                                               | 更新后的 DEM                           |
| `implicit_diffusion2D(topo)`                                                                                                                         | 逐行/逐列解三对角的隐式扩散（`scipy.linalg.solve_banded`），边界镜像处理。 | `topo`: 当前 DEM                                                                                                                                                                                             | 更新后的 DEM                           |
| `crank_nicolson2D(grid_old)`                                                                                                                         | CN 半隐式：`0.5*(explicit + implicit)`。                 | `grid_old`: 当前 DEM                                                                                                                                                                                         | 更新后的 DEM                           |
| `calc_scale_factor(diameter, radius, res, primary_index, continuous_ejecta_blanket_factor, X_grid, Y_grid, ones_grid, grid_size)`                    | 计算体量守恒的缩放系数：使“凹坑负体积”与“喷出/堆积正体积”在离散栅格上尽量平衡。          | `diameter/radius`: 坑径/半径 m；`res`: m/px；`primary_index`: 主/次坑调节因子；`continuous_ejecta_blanket_factor`: 喷出毯半径系数；`X_grid/Y_grid`: 网格坐标（`ogrid`）；`ones_grid`: 全 1；`grid_size`: px                               | `float`（>0）                        |
| `add_crater(grid, x_center, y_center, diameter, radius, res, grid_size, primary_index, continuous_ejecta_blanket_factor, X_grid, Y_grid, ones_grid)` | 将“坑体 + 连续喷出毯”剖面叠加到 DEM，含继承参数与体量守恒缩放；返回更新后的 DEM。     | `grid`: DEM；`x_center/y_center`: 像素坐标；`diameter/radius`: m（半径通常 = 坑径/2）；`res`: m/px；`grid_size`: px；`primary_index`: 主/次坑调节；`continuous_ejecta_blanket_factor`: 喷出毯半径系数；`X_grid/Y_grid/ones_grid`: 掩膜/权重辅助 | 更新后的 DEM                           |

---

### 4. ImpactorPopulation.py API 参考手册

| 方法名                                                                                                                                                                                                                                                         | 功能描述                                     | 参数说明                                                                                                                                                                                                                                                                                                                                               | 返回值                                                                                 |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `save_craters(d_craters, x_craters, y_craters, index_craters, t_craters)`                                                                                                                                                                                   | 写出事件清单为 CSV。                             | 列表/数组：`d_craters` 坑径 m；`x_craters/y_craters` 像素坐标；`index_craters` 类型（本项目 1=主坑）；`t_craters` 时间步                                                                                                                                                                                                                                                     | 无                                                                                   |
| `class ImpactorPopulation()`                                                                                                                                                                                                                                | 事件采样器（读取 `params` 的分布与常量）。               | 无                                                                                                                                                                                                                                                                                                                                                  | 实例                                                                                  |
| `calc_min_impactor(min_crater, v_i, theta_i)`                                                                                                                                                                                                               | 由最小可分辨坑径反推最小微陨石直径（π 组缩尺，返回 **km**）。      | `min_crater`: m；`v_i`: m/s；`theta_i`: 弧度（相对水平）                                                                                                                                                                                                                                                                                                     | `float`（km）                                                                         |
| `pi_group_scale_small(d_i, v_i, theta_i, frac_depth)`                                                                                                                                                                                                       | 粘土/风化层参数：由微陨石直径推最终坑径（米）。                 | `d_i`: m；`v_i`: m/s；`theta_i`: 弧度；`frac_depth`: 有效断裂/风化层厚度 m                                                                                                                                                                                                                                                                                       | `float`（m）                                                                          |
| `pi_group_scale_large(d_i, v_i, theta_i, frac_depth)`                                                                                                                                                                                                       | 岩石/基岩参数：由微陨石直径推最终坑径（米）。                  | 同上                                                                                                                                                                                                                                                                                                                                                 | `float`（m）                                                                          |
| `sample_impact_velocities(n_samples=1)`                                                                                                                                                                                                                     | 依据速度 CDF 双层随机采样速度（**km/s**；后续缩尺前会转 m/s）。 | `n_samples`: 数量                                                                                                                                                                                                                                                                                                                                    | `np.ndarray`（km/s）                                                                  |
| `sample_impact_angles(n_samples=1)`                                                                                                                                                                                                                         | 依据角度 CDF 采样入射角（弧度，分布 `p(θ)=½ sin2θ`）。    | `n_samples`: 数量                                                                                                                                                                                                                                                                                                                                    | `np.ndarray`（弧度）                                                                    |
| `sample_timestep_craters(t, avg_imp_diam, primary_lams, max_grid_dist, avg_crater_diam, num_inc, dt, min_crater, resolution, grid_size, grid_width, min_primary_for_secondaries, secondaries_on, r_body, continuous_ejecta_blanket_factor, X_grid, Y_grid)` | 在单个时间步内按各分段泊松采样主坑，并生成落点像素与坑径。            | `t`: 时间步；`avg_imp_diam`: 分段平均微陨石直径 **km**；`primary_lams`: 各分段 λ；`max_grid_dist`: 作用半径 m；`avg_crater_diam`: 代表坑径 m（参考）；`num_inc`: 分段面密度 `/m²/Myr`；`dt`: yr；`min_crater`: m；`resolution`: m/px；`grid_size`: px；`grid_width`: m；`min_primary_for_secondaries/secondaries_on/r_body`: 预留；`continuous_ejecta_blanket_factor`: 喷出毯系数；`X_grid/Y_grid`: 辅助 | `timestep_diams, timestep_x, timestep_y, timestep_primary_index, timestep_time`（数组） |
| `sample_all_craters()`                                                                                                                                                                                                                                      | 迭代全时段，汇总所有时间步事件清单。                       | 无                                                                                                                                                                                                                                                                                                                                                  | `d_craters, x_craters, y_craters, index_craters, t_craters`                         |

---

### 5. Tracers.py API 参考手册

| 方法名                                                                                                                             | 功能描述                                                                         | 参数说明                                                                      | 返回值              |
| ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------- |
| `class Tracer(x_p0, y_p0, z_p0)`                                                                                                | 单个示踪粒子；记录当前位置与历史轨迹。                                                          | 初始像素/高程：`x_p0, y_p0, z_p0`                                                | 实例               |
| `current_position()`                                                                                                            | 获取 `[x, y, z]` 当前位置。                                                         | 无                                                                         | `[x, y, z]`      |
| `update_position(new_position)`                                                                                                 | 写回新位置。                                                                       | `new_position=[x,y,z]`                                                    | 无                |
| `update_trajectory(x_p, y_p, z_p, d_p, slope_p)`                                                                                | 记录轨迹、深度与坡度。                                                                  | 粒子位置/深度/坡度                                                                | 无                |
| `tracer_particle_diffusion(grid_old, grid_new, particle_pos)`                                                                   | 扩散过程下的粒子更新：边界/NaN 特殊处理；若“负深度”则按 3×3 最陡下坡移动 1 像素（越界可周期回填）。                    | `grid_old/grid_new`: DEM；`particle_pos`: `[x,y,z]`                        | 更新后的 `[x, y, z]` |
| `solve_for_landing_point(t, *args)`                                                                                             | 给定飞行时间 `t`，返回“粒子高度 − 新表面高度”的绝对差（供 L-BFGS-B 最小化求着陆时刻）。                        | `*args=(R0, v_vert, x_c, y_c, phi0, f_surf_new, z0)`                      | `float`（残差）      |
| `solve_for_ejection_point(t, *args)`                                                                                            | 给定地下流动时间 `t`，返回“流线高度 − 旧表面高度”的绝对差（求出露时刻）。                                    | `*args=(R0, x_c, y_c, phi0, theta0, alpha, f_surf_old)`                   | `float`（残差）      |
| `tracer_particle_crater(x_p0, y_p0, z_p0, d_p0, dx, dy, dz, x_crater_pix, y_crater_pix, R0, crater_radius, grid_old, grid_new)` | 陨击事件对粒子的更新（表面湮灭/弹射/填充/掩埋；地下喷射/流动/钻孔等多情景），含越界周期回填。                            | 初始粒子状态与相对坑心的三维量：`d_p0, dx, dy, dz, R0`；坑心像素 `x/y`；`crater_radius`；旧/新 DEM | 更新后的 `[x, y, z]` |
| `sample_noise_val()`                                                                                                            | 采样亚像素“噪声”位移（Johnson SU 分布；**如需调用请在工程入口导入 `from scipy import stats as st`**）。 | 无                                                                         | `float`          |
| `tracer_particle_noise(x_p0, y_p0, z_p0, grid_old, noise)`                                                                      | 仅按噪声改变粒子 z（若超出表面则贴回表面），不做水平位移。                                               | 初始位置；`grid_old`；`noise` 正/负代表填充/挖掘                                        | 更新后的 `[x, y, z]` |
| `build_tracers_group()`                                                                                                         | 生成三维规则粒子群：水平 `n×n`（避开边界 5 像素），垂向 100 个层位（顶层 0）。                              | 无                                                                         | `list[Tracer]`   |

---

### 6. grain_3D.py API 参考手册（3D 曝露龄演示）

| 方法名                                                 | 功能描述                           | 参数说明                                | 返回值                   |
| --------------------------------------------------- | ------------------------------ | ----------------------------------- | --------------------- |
| `class SphereVoxel(position)`                       | 球面体素（存位置与曝露龄）。                 | `position`: `[x,y,z]`（单位球）          | 实例                    |
| `get_position()` / `refresh_position(new_position)` | 取/改体素位置。                       | `new_position`: `[x,y,z]`           | 位置或无                  |
| `get_exposure_age()` / `set_exposure_age(add_age)`  | 取/累加曝露龄。                       | `add_age`: 标量                       | 曝露龄或无                 |
| `generate_sphere_group(n_points, radius=1.0)`       | 按“黄金角”近似均匀地在球面布点并生成体素组。        | `n_points`: 个数；`radius`: 半径（默认 1.0） | `list[SphereVoxel]`   |
| `random_rotation_matrix()`                          | 随机生成 3×3 旋转矩阵。                 | 无                                   | `np.ndarray(3,3)`     |
| `rotate_point_cloud(group)`                         | 原地旋转体素点云。                      | `group`: 体素组                        | 无                     |
| `grain_exposure(group, add_age, h_thro=0)`          | 对 `z>h_thro` 的体素累加曝露龄（简单阈值面）。  | `add_age`: 标量；`h_thro`: 平面阈值        | 无                     |
| `grain_exposure_consider_self(group, add_age)`      | 以近似 `sinθ≈z/R` 加权累加曝露龄（自遮蔽效应）。 | `add_age`: 标量                       | 无                     |
| `get_average_exposure_age(group)`                   | 计算平均曝露龄。                       | `group`: 体素组                        | `float`               |
| `plot_colored_point_cloud(group)`                   | 用曝露龄着色绘制 3D 点云。                | `group`: 体素组                        | Matplotlib Figure（弹窗） |

> **可视化“1/4 修正”**：出图阶段可将统计曝露龄除以 4 作为遮蔽效应经验修正（参见脚本注释）。

---

## 可视化

可在 `display/` 中调用脚本进行展示：

* `draw_impact_dem.py`：陨击序列与 DEM 演化
* `draw_track.py`：粒子运动轨迹与空间分布
* `draw_sample_grains.py`：给定深度窗口内样品/粒子曝露龄与来源统计

如需演示 3D 粒子曝露龄修正（÷4 的遮蔽因子），直接：

```bash
python grain_3D.py
```

> “÷4”修正是在可视化阶段对模拟曝露龄作的折算，反映遮蔽效应；其由 `grain_3D.py` 的三维粒子仿真总结得到，并在绘图时统一处理。

---
## 示例数据

可选演示数据见文档提供的网盘地址（按需下载后放入本仓库display_demo文件内或自定义路径）：
`https://pan.baidu.com/s/1Egr4hF7ujG7UdC0brGh2fA?pwd=1234` 

---

## 贡献与交流

* 欢迎提交 issue、pull request 或通过邮件联系作者

---

## 许可协议

本项目基于 [MIT License](LICENSE) 开源。
请自由使用、修改和分发。

---

## 参考与致谢

本代码**部分基于** O’Brien 等人的开源模型进行修改与扩展，用于 CE-6 远端登陆区的小尺度模拟；如用于研究，请同时引用原始工作与相关 CE-6 论文。上述背景与用途说明见项目介绍文档。

---

## 代码维护与联系

**主要维护人员：**

* GitHub: [Henry-Baiheng](https://github.com/Henry-Baiheng)
* 邮箱: [1243244190@qq.com](mailto:1243244190@qq.com)

**作者与论文仓库：**

* GitHub: [boywc](https://github.com/boywc)
* 邮箱: [liurenruis@gmail.com](mailto:liurenruis@gmail.com)
* 项目主页: [https://github.com/boywc/Small-Scale-Impact-Gardening-Simulation-Model](https://github.com/boywc/Small-Scale-Impact-Gardening-Simulation-Model)
* 论文链接：[待公开](待公开)
* Git 克隆地址: `https://github.com/boywc/Small-Scale-Impact-Gardening-Simulation-Model.git`

---

**欢迎 Star、Fork 与使用本项目！如有问题或建议，欢迎通过 GitHub Issue 或邮件联系作者或维护者。**

---