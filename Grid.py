import params
import numpy as np
from scipy.interpolate import interp1d
import scipy

class Grid:
	def __init__(self, grid_size, resolution, diffusivity, dt):
		# 500, 4, 0.109, 1
		# Initialize grid parameters
		self.grid_size = grid_size             # 500
		self.resolution = resolution           # 4
		self.grid_width = grid_size*resolution # 2000

		if params.diffusion_on:
			def tridiag(a, b, c, k1=-1, k2=0, k3=1):
				# np.diag(a, k1) a为对角线元素 k1是对角线往下一行的对角线
				# 输出方阵，对角线下一行的对角线元素为列表 a
				return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

			R = 2*(resolution*resolution)/(diffusivity*dt) # 293.57798165137615
			a = -1.0*np.ones(grid_size-1)                  # shape (499,) val=-1
			c = -1.0*np.ones(grid_size-1)                  # shape (499,) val=-1
			b = (R + 2.0)*np.ones(grid_size)               # shape (500,) val=295.57798165
			b[0] = 1.0
			c[0] = 0.0                                     # [0,    -1, ...,     -1, -1] shape(499,)
			b[-1] = 1.0                                    # [1, 295.57, ..., 295.57, 1] shape(500,)
			a[-1] = 0.0                                    # [-1,    -1, ...,     -1, 0] shape(499,)
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
				n = a.shape[1]                     # 500
				assert(np.all(a.shape == (n, n)))  # 如果为真继续运行
				ab = np.zeros((2*n-1, n))          # (999, 500)

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

			self.R = R     # 293.57798165137615
			self.ab = ab   # shape (3, 500)

	def setUpGrid(self):
		return np.zeros((self.grid_size, self.grid_size))

	def implicit_diffusion2D(self, topo):
		R = self.R
		ab = self.ab
		grid_size = params.grid_size

		topo_inter = np.copy(topo)
		for i in range(1, grid_size-1):
			b1 = [topo[i,j] if j==0 or j==(grid_size-1) else (topo[i,j-1] + topo[i,j+1]) + (R-2.0)*topo[i,j] for j in range(grid_size)]

			u1 = scipy.linalg.solve_banded((1,1), np.copy(ab), b1, overwrite_ab=True, overwrite_b=True)
			topo_inter[i,:] = u1

		for j in range(1, grid_size-1):
			b2 = [topo[i,j] if i==0 or i==(grid_size-1) else (topo_inter[i-1,j] + topo_inter[i+1,j]) + (R-2.0)*topo_inter[i,j] for i in range(grid_size)]
			u2 = scipy.linalg.solve_banded((1,1), np.copy(ab), b2, overwrite_ab=True, overwrite_b=True)

			topo_inter[:,j] = u2

		topo_inter[0,:] = topo_inter[1,:]
		topo_inter[-1,:] = topo_inter[-2,:]
		topo_inter[:,0] = topo_inter[:,1]
		topo_inter[:,-1] = topo_inter[:,-2]

		return topo_inter

	def explicit_diffusion2D(self, u0):
		# Propagate with forward-difference in time, central-difference in space
		# Compute diffusion for grid using input parameters
		dx2 = params.dx2
		dy2 = params.dy2
		D = params.diffusivity
		dt = params.dt
		u = np.copy(u0)
		u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * ( (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2+ (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2 )

		u[0,:] = u[1,:]
		u[-1,:] = u[-2,:]
		u[:,0] = u[:,1]
		u[:,-1] = u[:,-2]

		return u

	def crank_nicolson2D(self, grid_old):
		exp_grid = self.explicit_diffusion2D(grid_old)
		imp_grid = self.implicit_diffusion2D(grid_old)

		cn_grid = 0.5*exp_grid + 0.5*imp_grid

		return cn_grid

	def calc_scale_factor(self, diameter, radius, res, primary_index, continuous_ejecta_blanket_factor, X_grid, Y_grid, ones_grid, grid_size):

		x_crater = y_crater = grid_size/2

		#depth = (269.0/81.0)*(0.04*diameter)*primary_index
		#rim_height = 0.04*diameter*primary_index

		depth = 0.2*diameter*primary_index
		rim_height = ((3.0*continuous_ejecta_blanket_factor**3)/(15.0*continuous_ejecta_blanket_factor**3 - 16.0*continuous_ejecta_blanket_factor**2 + 2.0*continuous_ejecta_blanket_factor + 2.0))*depth


		r_ejecta = continuous_ejecta_blanket_factor*radius

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

		scaling_factor = np.sum(crater_grid[crater_grid >= 0.0])/(abs(np.sum(crater_grid[crater_grid < 0.0])))

		return scaling_factor

	def add_crater(self, grid, x_center, y_center, diameter, radius, res, grid_size, primary_index, continuous_ejecta_blanket_factor, X_grid, Y_grid, ones_grid):

		radius = diameter/2.0
		r_ejecta = continuous_ejecta_blanket_factor*radius # 溅射毯范围
		# 网格距离撞击坑的距离
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
			# 计算 rim 高度
			rim_height = ((3.0*continuous_ejecta_blanket_factor**3)/(15.0*continuous_ejecta_blanket_factor**3 - 16.0*continuous_ejecta_blanket_factor**2 + 2.0*continuous_ejecta_blanket_factor + 2.0))*depth
			# 撞击坑内像素
			full_crater_mask = dist_from_center <= r_ejecta
			# Crater somewhat overlaps the grid
			# Inheritance parameter, set as a constant, see Howard 2007, I=0 --> crater rim horizontal, I=1 --> crater rim parallel to pre-existing slope
			I_i = 0.9

			# Reference elevation is the average of the pre-crater grid within the crater area
			# Creter interior weighted by 1, ejecta blanket weighted by (distance/radius)**-n, n=3
			interior_mask = dist_from_center <= radius  # 撞击坑内
			exterior_mask = dist_from_center > radius   # 撞击坑外

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

			crater_inh_profile = (E_r - grid)*(1.0 - (I_i*(dist_from_center/radius)**2))
			ejecta_inh_profile = G_grid*(E_r - grid)

			delta_H_crater +=crater_inh_profile
			delta_H_ejecta += ejecta_inh_profile

			# Add calculated elevations to the grid at the corresponding pixels、
			# 将计算出的高程添加到相应像素的网格中
			crater_grid = np.zeros((grid_size, grid_size))
			crater_grid[crater_mask] = delta_H_crater[crater_mask]
			crater_grid[ejecta_mask] = delta_H_ejecta[ejecta_mask]

			scaling_factor = [self.calc_scale_factor(diameter, radius, res, primary_index, continuous_ejecta_blanket_factor, X_grid, Y_grid, ones_grid, grid_size) if diameter <= (grid_size*res/3.0) else 1.0]

			crater_grid[crater_grid < 0.0] *= scaling_factor

			grid += crater_grid

		return grid