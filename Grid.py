"""
Grid：二维高度场网格与扩散/陨击坑形貌模型

本文件提供：
1) 网格初始化与若干数值扩散方案：
   - 显式(Explicit)中心差分
   - 近似隐式(Implicit)按行、按列分步三对角解法（带状矩阵形式，scipy.linalg.solve_banded）
   - Crank–Nicolson 半隐式（显式与隐式平均）

2) 陨击坑（含连续喷出物毯）叠加模型：
   - 根据坑径/深度/边缘(rim)高度与喷出毯范围构造高程剖面
   - 继承(Inheritance)机制融入原始地形
   - 体量守恒尺度因子（正负体积匹配）计算

依赖外部参数模块 `params`：
- params.diffusion_on: 是否启用隐式扩散所需的预处理
- params.grid_size, params.dx2, params.dy2, params.diffusivity, params.dt: 网格与扩散参数


"""

import params
import numpy as np
from scipy.interpolate import interp1d
import scipy


class Grid:
	"""
	二维网格容器与数值模型。

	Args:
		grid_size (int): 网格点数（每边），例：500。
		resolution (float): 单元分辨率（每像素或网格单位长度），例：4。
		diffusivity (float): 扩散系数 D。
		dt (float): 时间步长 Δt。

	Attributes:
		grid_size (int): 同入参。
		resolution (float): 同入参。
		grid_width (float): 物理宽度 = grid_size * resolution。
		R (float): 隐式离散式中的参数 2Δx⁻² / (DΔt) 的等价组合（见下式构造）。
		ab (ndarray): 三对角矩阵的带状存储格式，用于 solve_banded。
	"""

	def __init__(self, grid_size, resolution, diffusivity, dt):
		# 例：500, 4, 0.109, 1
		# Initialize grid parameters
		self.grid_size = grid_size             # 500
		self.resolution = resolution           # 4
		self.grid_width = grid_size*resolution # 2000

		# 若开启隐式扩散，需要预构造行/列方向的三对角系数矩阵（并转为带状格式）
		if params.diffusion_on:
			def tridiag(a, b, c, k1=-1, k2=0, k3=1):
				"""
				构造标准三对角方阵：
				- 下对角线：np.diag(a, k=-1)
				- 主对角线：np.diag(b, k=0)
				- 上对角线：np.diag(c, k=+1)
				"""
				# np.diag(a, k1) a为对角线元素，k1是对角线往下一行的对角线
				# 输出方阵，对角线下一行的对角线元素为列表 a
				return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

			# R 来自二维扩散方程隐式离散系数的归一化组合（行/列分步解）
			# 这里将空间步长用 resolution 表示，数值上相当于 Δx = resolution
			R = 2*(resolution*resolution)/(diffusivity*dt) # 293.57798165137615

			# 组装三对角系数：A * u_new = RHS
			# 中间网格点：主对角 (R+2)，上下对角 -1；边界行用 1 处理（相当于单位行）
			a = -1.0*np.ones(grid_size-1)                  # shape (499,) val=-1
			c = -1.0*np.ones(grid_size-1)                  # shape (499,) val=-1
			b = (R + 2.0)*np.ones(grid_size)               # shape (500,) val=295.57798165
			b[0] = 1.0
			c[0] = 0.0                                     # [0,    -1, ...,     -1, -1] shape(499,)
			b[-1] = 1.0                                    # [1, 295.57, ..., 295.57, 1] shape(500,)
			a[-1] = 0.0                                    # [-1,    -1, ...,     -1, 0] shape(499,)

			# 临时占位（未直接使用），保留以保持与原始代码一致
			ab = np.zeros((3, grid_size-2))                # shape (3, 498) val=0

			A = tridiag(a, b, c)
			# A = shape (500, 500)
			# [[1.    0.       0.      ...     0.       0.	     0.]
			#  [-1.   295.57,  - 1.    ...     0.       0.	     0.]
			#  [0.    - 1.	   295.57  ...	   0.	    0.       0.]
			#   ...
			#  [0. 	  0.       0.      ...	   295.57  -1. 	     0.]
			#  [0.    0.       0.      ...     -1.      295.57  -1.]
			#  [0.	  0.       0.      ...     0.       0.       1.]]

			def diagonal_form(a, upper = 1, lower= 1):
				"""
				将稠密三对角方阵 a 转换为 `solve_banded` 需要的带状格式。
				仅保留上下各 `upper/lower` 条带与主对角线。

				Args:
					a (ndarray): n×n 方阵。
					upper (int): 上带宽（对角条数）。
					lower (int): 下带宽。

				Returns:
					ndarray: 形状为 ((upper+lower+1), n) 的带状矩阵。
				"""
				n = a.shape[1]                     # 500
				assert(np.all(a.shape == (n, n)))  # 如果为真继续运行
				ab = np.zeros((2*n-1, n))          # (999, 500)

				# 逐条抽取对角线到中间层，再剪裁为需要的 3 行
				for i in range(n):  # i:0-499
					ab[i, (n-1)-i:] = np.diagonal(a, (n-1)-i)
				for i in range(n-1):
					ab[(2*n-2)-i, :i+1] = np.diagonal(a, i-(n-1))

				mid_row_inx = int(ab.shape[0]/2)   # 499
				upper_rows = [mid_row_inx - i for i in range(1, upper+1)] # [498]
				upper_rows.reverse()               # 列表中数据的反转
				upper_rows.append(mid_row_inx)
				lower_rows = [mid_row_inx + i for i in range(1, lower+1)]
				keep_rows = upper_rows+lower_rows
				ab = ab[keep_rows, :] # 构建一个带状矩阵
				# ab = (3, 500)
				# [[0.     0.        -1.      ...   -1.      -1.        -1.]
				#  [1.     295.57    295.577  ...   295.57   295.57      1.]
				#  [-1.    -1.       -1.      ...   -1.      0.          0.]]
				return ab

			ab = diagonal_form(A)

			# 保存隐式求解需要的系数
			self.R = R     # 293.57798165137615
			self.ab = ab   # shape (3, 500)

	def setUpGrid(self):
		"""
		创建并返回一个全零高度场。

		Returns:
			ndarray: 形状 (grid_size, grid_size) 的零矩阵。
		"""
		return np.zeros((self.grid_size, self.grid_size))

	def implicit_diffusion2D(self, topo):
		"""
		二维隐式扩散（逐行、逐列分步三对角解；近似 ADI 思想）。

		思路：
		    1) 固定列、对每一“行”构造 RHS，调用 `solve_banded` 解三对角；
		    2) 固定行、对每一“列”再解一次；
		   边界采用与原代码一致的“邻值复制”（Neumann 型近似）处理。

		Args:
		    topo (ndarray): 上一时刻/当前时刻高度场（将被平滑）。

		Returns:
		    ndarray: 扩散后的高度场（同形状）。
		"""
		R = self.R
		ab = self.ab
		grid_size = params.grid_size

		topo_inter = np.copy(topo)

		# 第一次：沿行方向解三对角
		for i in range(1, grid_size-1):
			# 内点: u_{j-1} + (R-2) u_j + u_{j+1}
			# 边界: 直接使用原值（b1[j] = topo[i,j]）
			b1 = [topo[i,j] if j==0 or j==(grid_size-1) else (topo[i,j-1] + topo[i,j+1]) + (R-2.0)*topo[i,j] for j in range(grid_size)]

			u1 = scipy.linalg.solve_banded((1,1), np.copy(ab), b1, overwrite_ab=True, overwrite_b=True)
			topo_inter[i,:] = u1

		# 第二次：沿列方向解三对角
		for j in range(1, grid_size-1):
			b2 = [topo[i,j] if i==0 or i==(grid_size-1) else (topo_inter[i-1,j] + topo_inter[i+1,j]) + (R-2.0)*topo_inter[i,j] for i in range(grid_size)]
			u2 = scipy.linalg.solve_banded((1,1), np.copy(ab), b2, overwrite_ab=True, overwrite_b=True)

			topo_inter[:,j] = u2

		# 边界条件：邻值复制（近似零法向梯度）
		topo_inter[0,:] = topo_inter[1,:]
		topo_inter[-1,:] = topo_inter[-2,:]
		topo_inter[:,0] = topo_inter[:,1]
		topo_inter[:,-1] = topo_inter[:,-2]

		return topo_inter

	def explicit_diffusion2D(self, u0):
		"""
		二维显式扩散：前向时间、中心空间差分（FTCS）。

		离散式：
		    u^{n+1}_{i,j} = u^{n}_{i,j} + DΔt [ (u_{i+1,j}-2u_{i,j}+u_{i-1,j})/Δx^2
		                                       + (u_{i,j+1}-2u_{i,j}+u_{i,j-1})/Δy^2 ]
		稳定性提示：显式格式需满足经典 CFL 条件。

		Args:
		    u0 (ndarray): 当前高度场。

		Returns:
		    ndarray: 单步扩散后的高度场。
		"""
		# Propagate with forward-difference in time, central-difference in space
		# Compute diffusion for grid using input parameters
		dx2 = params.dx2
		dy2 = params.dy2
		D = params.diffusivity
		dt = params.dt
		u = np.copy(u0)
		u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * ( (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2+ (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2 )

		# 边界：邻值复制
		u[0,:] = u[1,:]
		u[-1,:] = u[-2,:]
		u[:,0] = u[:,1]
		u[:,-1] = u[:,-2]

		return u

	def crank_nicolson2D(self, grid_old):
		"""
		Crank–Nicolson 半隐式：显式与隐式各占 0.5 权重的线性组合。

		Args:
		    grid_old (ndarray): 当前高度场。

		Returns:
		    ndarray: 单步 Crank–Nicolson 结果。
		"""
		exp_grid = self.explicit_diffusion2D(grid_old)
		imp_grid = self.implicit_diffusion2D(grid_old)

		cn_grid = 0.5*exp_grid + 0.5*imp_grid

		return cn_grid

	def calc_scale_factor(self, diameter, radius, res, primary_index, continuous_ejecta_blanket_factor, X_grid, Y_grid, ones_grid, grid_size):
		"""
		计算体量守恒尺度因子，用于“坑内凹陷（负体积）与喷出/堆积（正体积）”匹配。

		思路：
		    - 构造坑体与喷出毯的高程剖面，生成临时网格 crater_grid
		    - 统计 crater_grid 中的正体积与负体积
		    - scale = sum(positive) / abs(sum(negative))
		    - 后续将负值乘以该 scale，使二者尽量平衡

		注：
		    - 本函数只返回比例因子，不修改原网格。
		    - depth 与 rim_height 采用与 add_crater 相同的解析式。

		Returns:
		    float: 负体积缩放系数（>0）。
		"""
		x_crater = y_crater = grid_size/2

		#depth = (269.0/81.0)*(0.04*diameter)*primary_index
		#rim_height = 0.04*diameter*primary_index

		depth = 0.2*diameter*primary_index
		rim_height = ((3.0*continuous_ejecta_blanket_factor**3)/(15.0*continuous_ejecta_blanket_factor**3 - 16.0*continuous_ejecta_blanket_factor**2 + 2.0*continuous_ejecta_blanket_factor + 2.0))*depth

		r_ejecta = continuous_ejecta_blanket_factor*radius

		# 到坑中心的物理距离（单位：与 res 一致）
		dist_from_center = np.hypot(abs(X_grid - x_crater)*res, abs(Y_grid - y_crater)*res)

		# Grid pixels covered by the crater
		crater_mask = dist_from_center <= radius

		# Grid pixels covered by the ejecta blanket
		ejecta_mask = (dist_from_center > radius) & (dist_from_center <= r_ejecta)

		# Crater elevation profile
		delta_H_crater = (((dist_from_center/radius)**2)*(rim_height + depth)) - depth

		# Ejecta elevation profile
		# Divide by zero at r=0 but we don't care about that point since it's interior to the ejecta blanket
		with np.errstate(divide='ignore'):
			delta_H_ejecta = rim_height*((dist_from_center/radius)**-3) - (rim_height/(continuous_ejecta_blanket_factor**3*(continuous_ejecta_blanket_factor - 1.0)))*((dist_from_center/radius) - 1.0)

		crater_grid = np.zeros((grid_size, grid_size))
		crater_grid[crater_mask] = delta_H_crater[crater_mask]
		crater_grid[ejecta_mask] = delta_H_ejecta[ejecta_mask]

		# 正/负体积比值
		scaling_factor = np.sum(crater_grid[crater_grid >= 0.0])/(abs(np.sum(crater_grid[crater_grid < 0.0])))

		return scaling_factor

	def add_crater(self, grid, x_center, y_center, diameter, radius, res, grid_size, primary_index, continuous_ejecta_blanket_factor, X_grid, Y_grid, ones_grid):
		"""
		在现有高度场 `grid` 上叠加一个环形坑与其连续喷出物毯。

		参数含义（与原代码一致）：
		    x_center, y_center (float): 坑中心在网格坐标系下的位置（单位：像素格）。
		    diameter (float): 坑直径。
		    radius (float): 坑半径（通常 = diameter/2，会在函数内重赋值）。
		    res (float): 网格分辨率（长度/像素）。
		    grid_size (int): 网格边长（像素数）。
		    primary_index (float): 主/次生坑调节因子（影响深度等）。
		    continuous_ejecta_blanket_factor (float): 连续喷出物毯范围系数（r_ejecta = f*radius）。
		    X_grid, Y_grid (ndarray): 与 `grid` 同形的坐标网格。
		    ones_grid (ndarray): 与 `grid` 同形的全 1 矩阵，用于加权/继承。
		流程：
		    1) 计算坑内与喷出毯覆盖区域的掩码；
		    2) 用解析式给出坑深/边缘高度与喷出毯剖面；
		    3) 采用继承矩阵 G_grid 以及 I_i 融入原地形（参见 Howard 2007 的思想注释）；
		    4) 可选体量守恒：当坑径不超过域宽的 1/3 时，按正负体积比缩放凹陷；
		    5) 将结果累加到输入的 `grid` 上并返回。

		Returns:
		    ndarray: 已叠加坑形的高度场（原地形被原地修改后返回）。
		"""
		# 以传入 diameter 为准，radius 会被重赋值保持一致
		radius = diameter/2.0
		r_ejecta = continuous_ejecta_blanket_factor*radius # 溅射毯范围

		# 网格到撞击坑中心的物理距离（乘 res 转为物理尺度）
		dist_from_center = np.hypot(abs(X_grid - x_center)*res, abs(Y_grid - y_center)*res)

		# Grid pixels covered by the crater 陨石坑覆盖的网格像素
		crater_mask = dist_from_center <= radius

		# Grid pixels covered by the ejecta blanket 喷出毯覆盖+坑的网格像素
		ejecta_mask = (dist_from_center > radius) & (dist_from_center <= r_ejecta)

		if np.sum(crater_mask) == 0 and np.sum(ejecta_mask) == 0:
			# 火山口实际上没有与网格重叠。为了完整起见，这些情况也包括在内，只需返回火山口前的网格即可
			# Crater does not actually overlap the grid. These cases are included for completeness, simply return the pre-crater grid
			pass

		else:

			#depth = (269.0/81.0)*(0.04*diameter)*primary_index
			#rim_height = 0.04*diameter*primary_index

			depth = 0.2*diameter*primary_index # 深度=0.2*直径*是否是次级坑

			# 计算 rim 高度（与 calc_scale_factor 中一致）
			rim_height = ((3.0*continuous_ejecta_blanket_factor**3)/(15.0*continuous_ejecta_blanket_factor**3 - 16.0*continuous_ejecta_blanket_factor**2 + 2.0*continuous_ejecta_blanket_factor + 2.0))*depth

			# 撞击坑内像素（含边缘与喷出毯范围）
			full_crater_mask = dist_from_center <= r_ejecta

			# Crater somewhat overlaps the grid
			# Inheritance parameter, set as a constant, see Howard 2007, I=0 --> crater rim horizontal, I=1 --> crater rim parallel to pre-existing slope
			I_i = 0.9

			# Reference elevation is the average of the pre-crater grid within the crater area
			# Creter interior weighted by 1, ejecta blanket weighted by (distance/radius)**-n, n=3
			interior_mask = dist_from_center <= radius  # 撞击坑内
			exterior_mask = dist_from_center > radius   # 撞击坑外

			# 加权参考高程 E_r：坑内权重 1；坑外（喷出毯）权重 (r/radius)^(-3)
			weights = np.copy(ones_grid)
			weights[exterior_mask] = (dist_from_center[exterior_mask]/radius)**-3
			weighted_grid = grid*weights
			E_r = np.average(grid[full_crater_mask], weights=weights[full_crater_mask])

			# Crater elevation profile 火山口高程剖面图
			delta_H_crater = (((dist_from_center/radius)**2)*(rim_height + depth)) - depth

			# Ejecta elevation profile 喷出物高程剖面图
			# Divide by zero at r=0 but we don't care about that point since it's interior to the ejecta blanket
			# 在 r=0 处除以零，但我们并不关心该点，因为它位于抛射物毯内部
			with np.errstate(divide='ignore'):
				delta_H_ejecta = rim_height*((dist_from_center/radius)**-3) - (rim_height/(continuous_ejecta_blanket_factor**3*(continuous_ejecta_blanket_factor - 1.0)))*((dist_from_center/radius) - 1.0)

			# Inheritance matrices - determines how the crater is integrated into the existing grid
			# 继承矩阵--决定如何将火山口整合到现有网格中
			G_grid = (1.0 - I_i)*ones_grid
			min_mask = G_grid > delta_H_ejecta/rim_height
			G_grid[min_mask] = delta_H_ejecta[min_mask]/rim_height

			# 坑内/喷出毯的继承剖面（随距离衰减或受限于 G_grid）
			crater_inh_profile = (E_r - grid)*(1.0 - (I_i*(dist_from_center/radius)**2))
			ejecta_inh_profile = G_grid*(E_r - grid)

			# 将继承项叠加到解析剖面上
			delta_H_crater +=crater_inh_profile
			delta_H_ejecta += ejecta_inh_profile

			# Add calculated elevations to the grid at the corresponding pixels、
			# 将计算出的高程添加到相应像素的网格中
			crater_grid = np.zeros((grid_size, grid_size))
			crater_grid[crater_mask] = delta_H_crater[crater_mask]
			crater_grid[ejecta_mask] = delta_H_ejecta[ejecta_mask]

			# 体量守恒：仅当坑径不超过域宽的 1/3 时启用（与原逻辑一致）
			scaling_factor = [self.calc_scale_factor(diameter, radius, res, primary_index, continuous_ejecta_blanket_factor, X_grid, Y_grid, ones_grid, grid_size) if diameter <= (grid_size*res/3.0) else 1.0]

			# 对凹陷（负值）进行缩放，使正负体积尽量平衡
			crater_grid[crater_grid < 0.0] *= scaling_factor

			# 将坑形累加进原地形（原地修改）
			grid += crater_grid

		return grid
