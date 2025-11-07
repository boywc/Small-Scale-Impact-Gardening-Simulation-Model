import params
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import interpolate
from scipy.optimize import minimize

class Tracer:
    def __init__(self, x_p0, y_p0, z_p0):
            # Initialize particle position with x,y,z position
            # Create arrays to store the particle's trajectory
        self.x_p = x_p0
        self.y_p = y_p0
        self.z_p = z_p0
        self.x_arr = []
        self.y_arr = []
        self.z_arr = []
        self.d_arr = []
        self.slope_arr = []

    def current_position(self):
        # Return the particle's current position
        # Should be used to get inputs for transportation methods
        return [self.x_p, self.y_p, self.z_p]

    def update_position(self, new_position):
        # Update particle's current position
        self.x_p = new_position[0]
        self.y_p = new_position[1]
        self.z_p = new_position[2]

    def update_trajectory(self, x_p, y_p, z_p, d_p, slope_p):
        # Update particle history at the end of each timestep
        # Store x,y,z position as well as depth and the slope of the grid at the particle's position
        self.x_arr.append(x_p)
        self.y_arr.append(y_p)
        self.z_arr.append(z_p)
        self.d_arr.append(d_p)
        self.slope_arr.append(slope_p)

    # 地形退化对颗粒追迹的影响
    def tracer_particle_diffusion(self, grid_old, grid_new, particle_pos):
        diffusivity = params.diffusivity
        dt = params.dt
        dx2 = params.dx2
        dy2 = params.dy2
        grid_size = params.grid_size
        resolution = params.resolution

        if np.isnan(particle_pos[0]):
                # CLASS 0 - NaN particle passed to function
                # Should not happen, but if it does just return NaNs again
            x_p = np.nan
            y_p = np.nan
            z_p = np.nan
            print('NaN particle passed to function')
        # CLASS 1 - 粒子从网格边缘开始运动
        elif particle_pos[0] == 0 or particle_pos[0] == (grid_size-1) or particle_pos[1] == 0 or particle_pos[1] == (grid_size-1):
            # CLASS 1 - Particle starts on the edge of the grid
            # Cannot compute elevation difference in at least one direction so particle is lost
            # 无法计算至少一个方向的高差，因此粒子丢失
            # 如果是周期性粒子，则在与丢失粒子相同深度的随机位置添加一个新粒子
            # 如果没有，粒子就会消失在模型中

            # If periodic particles, add a new particle at a random position at the same depth as the lost particle
            # If not, particle is lost to the model
            if params.periodic_particles:
                x_p0 = int(round(particle_pos[0]))
                y_p0 = int(round(particle_pos[1]))
                z_p0 = particle_pos[2]
                d_p0 = grid_old[x_p0, y_p0] - z_p0

                x_p = int(np.random.randint(low=0, high=grid_size))
                y_p = int(np.random.randint(low=0, high=grid_size))
                # Place particle randomly on the grid at the particle's initial depth since we didn't compute a new position for it
                z_p = grid_new[x_p, y_p] - d_p0
            else:
                x_p = np.nan
                y_p = np.nan
                z_p = np.nan

        # # 第 2 类 - 可能通过扩散移动的粒子
        else:
            # CLASS 2 - Particle possibly moved by diffusion
            x_p = int(round(particle_pos[0]))
            y_p = int(round(particle_pos[1]))
            z_p = particle_pos[2]

            z_new = grid_new[x_p, y_p]
            z_old = grid_old[x_p, y_p]
            d_p0 = z_old - z_p
            # 情况 1 - 粒子在扩散引起的高度变化后仍在地表或地表以下。 粒子仍在原地
            if z_p < z_new:
                # SCENARIO 1 - Particle still at or below the surface after the elevation change from diffusion.  Particle remains where it is
                x_p = int(x_p)
                y_p = int(y_p)
                z_p = z_p
                # FINAL PARTICLE POSITION DETERMINED - C2:S1

            # 情况 2 - 粒子所在位置的扩散海拔高度变化会使其处于负深度。
            elif z_p >= z_new:

                # SCENARIO 2 - Diffusion elevation change at the particle's position would put it at negative depth.
                # Particle moves one pixel in the steepest downhill direction
                delta_h = z_old - z_new		# Change in elevation at the current pixel

                # 颗粒九宫格方向
                # Possible directions to move, I will be going from left to right then down to the next row
                # 可能的移动方向，我将从左到右，然后向下移动到下一排
                # (i-1, j+1)    (i, j+1)    (i+1, j+1)
                # (i-1, j)      (i, j)      (i+1, j)
                # (i-1, j-1)    (i, j-1)    (i+1, j-1)
                delta_z_11 = grid_old[x_p, y_p] - grid_old[x_p-1, y_p+1]
                delta_z_12 = grid_old[x_p, y_p] - grid_old[x_p, y_p+1]
                delta_z_13 = grid_old[x_p, y_p] - grid_old[x_p+1, y_p+1]
                delta_z_21 = grid_old[x_p, y_p] - grid_old[x_p-1, y_p]
                # trivial, particle must move somewhere other than the current pixel
                # 微不足道，粒子必须移动到当前像素以外的地方
                delta_z_22 = grid_old[x_p, y_p] - grid_old[x_p, y_p]
                delta_z_23 = grid_old[x_p, y_p] - grid_old[x_p+1, y_p]
                delta_z_31 = grid_old[x_p, y_p] - grid_old[x_p-1, y_p-1]
                delta_z_32 = grid_old[x_p, y_p] - grid_old[x_p, y_p-1]
                delta_z_33 = grid_old[x_p, y_p] - grid_old[x_p+1, y_p-1]

                delta_z = np.zeros((3, 3))
                delta_z[0, 0] = delta_z_11
                delta_z[0, 1] = delta_z_12
                delta_z[0, 2] = delta_z_13
                delta_z[1, 0] = delta_z_21
                delta_z[1, 1] = delta_z_22
                delta_z[1, 2] = delta_z_23
                delta_z[2, 0] = delta_z_31
                delta_z[2, 1] = delta_z_32
                delta_z[2, 2] = delta_z_33

                # 对角线方向用三角函数算一下
                rad_ext_11 = np.sqrt(2.0*resolution**2)
                rad_ext_12 = resolution
                rad_ext_13 = np.sqrt(2.0*resolution**2)
                rad_ext_21 = resolution
                rad_ext_22 = resolution
                rad_ext_23 = resolution
                rad_ext_31 = np.sqrt(2.0*resolution**2)
                rad_ext_32 = resolution
                rad_ext_33 = np.sqrt(2.0*resolution**2)

                slope_11 = np.arctan2(delta_z_11, rad_ext_11)
                slope_12 = np.arctan2(delta_z_12, rad_ext_12)
                slope_13 = np.arctan2(delta_z_13, rad_ext_13)
                slope_21 = np.arctan2(delta_z_21, rad_ext_21)
                slope_22 = 0
                slope_23 = np.arctan2(delta_z_23, rad_ext_23)
                slope_31 = np.arctan2(delta_z_31, rad_ext_31)
                slope_32 = np.arctan2(delta_z_32, rad_ext_32)
                slope_33 = np.arctan2(delta_z_33, rad_ext_33)

                x_change = [-1, 0, 1, -1, 0, 1, -1,  0,  1]
                y_change = [1, 1, 1,  0, 0, 0, -1, -1, -1]
                # 计算哪个方向梯度最大就向哪个方向移动
                max_slope_dir = np.argmax([slope_11, slope_12, slope_13, slope_21, slope_22, slope_23, slope_31, slope_32, slope_33], axis=0)

                x_p += x_change[max_slope_dir]
                y_p += y_change[max_slope_dir]
                # 如果粒子没滚出去
                if 0 <= x_p <= (grid_size-1) and 0 <= y_p <= (grid_size-1):
                        # Particle moves somewhere on the grid.  Placed at a depth correspondingly inversely to its initial depth (layers are flipped as material near the surface moves downslope first, followed by material buried deeper)
                    x_p = int(round(x_p))
                    y_p = int(round(y_p))
                    z_p = grid_new[x_p, y_p] - delta_h + d_p0 # 同时考虑了埋在地下的和在表面的两种情况，并且地形退化影响到了这个深度的粒子
                # 如果粒子滚出去了
                else:
                    # Particle diffuses off the grid

                    if params.periodic_particles:
                        d_p0 = z_old - z_p

                        x_p = int(np.random.randint(low=0, high=grid_size))
                        y_p = int(np.random.randint(low=0, high=grid_size))
                        # Place particle randomly on the grid at depth that depends inversely on its initial depth and on the amount of material that diffused from its initial pixel.
                        z_p = grid_new[x_p, y_p] - abs(-delta_h + d_p0)
                        # This is the same as we do for particles that diffuse on the grid, now assuming that the off-grid elevation is zero everywhere. z_p = 0 - delta_h + d_p0
                    else:
                        x_p = np.nan
                        y_p = np.nan
                        z_p = np.nan
                # FINAL PARTICLE POSITION DETERMINED - C2:S2

        d_p = grid_new[int(x_p), int(y_p)] - z_p
        if d_p < 0.0: # 如果地形退化没有影响到这个颗粒
            print('PARTICLE ABOVE THE SURFACE - END OF DIFFUSION FUNCTION')
            sys.exit()
            z_p = grid_new[x_p, y_p]

        return [x_p, y_p, z_p]

    def solve_for_landing_point(self, t, *args):
        resolution = params.resolution
        g = params.g

        R0 = args[0]
        ejection_velocity_vertical = args[1]
        x_crater_pix = args[2]
        y_crater_pix = args[3]
        phi0 = args[4]
        f_surf_new = args[5]
        z_p0 = args[6]

        r_t_flight = R0 + ejection_velocity_vertical*t
        z_t_flight = z_p0 + ejection_velocity_vertical*t - 0.5*g*(t**2)

        R_t = np.hypot(r_t_flight, z_t_flight)
        theta_t = np.arccos(z_t_flight/R_t)

        x_t_flight = R_t*np.sin(theta_t)*np.cos(phi0)/resolution + x_crater_pix
        y_t_flight = R_t*np.sin(theta_t)*np.sin(phi0)/resolution + y_crater_pix

        z_t_surf = f_surf_new(y_t_flight, x_t_flight)[0]

        return abs(z_t_surf - z_t_flight)

    def solve_for_ejection_point(self, t, *args):
        resolution = params.resolution

        R0 = args[0]
        x_crater_pix = args[1]
        y_crater_pix = args[2]
        phi0 = args[3]
        theta0 = args[4]
        alpha = args[5]
        f_surf_old = args[6]

        R_t_flow = (R0**4 + 4*alpha*t)**(1.0/4.0)
        theta_t_flow = np.arccos(1.0 - ((1.0-np.cos(theta0))*(R_t_flow/R0)))

        x_t_flow = (R_t_flow*np.sin(theta_t_flow)*np.cos(phi0))/resolution + x_crater_pix
        y_t_flow = (R_t_flow*np.sin(theta_t_flow)*np.sin(phi0))/resolution + y_crater_pix
        z_t_flow = -1.0*R_t_flow*np.cos(theta_t_flow)

        z_t_surf = f_surf_old(y_t_flow, x_t_flow)[0]

        return abs(z_t_surf - z_t_flow)

    def tracer_particle_crater(self, x_p0, y_p0, z_p0, d_p0, dx, dy, dz, x_crater_pix, y_crater_pix, R0, crater_radius, grid_old, grid_new):
        plot_on = 0
        print_on = 0

        grid_size = params.grid_size
        g = params.g
        continuous_ejecta_blanket_factor = params.continuous_ejecta_blanket_factor
        resolution = params.resolution

        if np.isnan(x_p0) or np.isnan(y_p0) or np.isnan(z_p0):
            ##### ------------------------------------------------------------------ #####
            # CLASS 0: NaN Particle passed.  Should not make it into the function but just return NaNs again if it does
            # CLASS 0：传入 NaN 粒子。 不应进入函数，如果进入则再次返回 NaNs
            print('CLASS 0')
            x_p = np.nan
            y_p = np.nan
            z_p = np.nan
            # FINAL PARTICLE POSITION DETERMINED - C0

        else:
            # Particle passed with real coordinates 以实数坐标传递的粒子
            # Define variables that will be used by all or most paths 定义所有或大部分路径都将使用的变量
            transient_crater_radius = crater_radius/1.18  # 瞬态坑半径
            alpha = np.sqrt((transient_crater_radius**7)*g/12.0)
            t_flow = (1.0/(4.0*alpha))*(12.0*(alpha**2)/g)**(4.0/7.0)   # Eq11

            # 第 1 类：粒子位于表面之上（非物理现象）
            if d_p0 < 0.0:
                ##### ------------------------------------------------------------------ #####
                # CLASS 1: Particle is above the surface (un-physical)
                # 第 1 类：粒子位于表面之上（非物理现象）
                print('CLASS 0: PARTICLE ABOVE SURFACE - BEGINNING OF CRATER FUNCTION')
                # FINAL PARTICLE POSITION DETERMINED - C1
                sys.exit()

            # 第 2 类：影响范围内表面上的粒子
            elif d_p0 == 0.0:
                ##### ------------------------------------------------------------------ #####
                # CLASS 2: Particle on the surface within the sphere of influence
                # 情况1： 判断点是否在撞击点，如果在，粒子湮灭，如果不在，粒子运动
                if R0 == 0.0: # 判断点是否在撞击点，如果在，粒子湮灭，如果不在，粒子运动
                    # SCENARIO 1: Obliteration 情景 1：湮灭
                    if print_on:
                        print('CLASS 2 - SCENARIO 1')
                    # Particle on the surface at the impact site and is obliterated 撞击点表面上的粒子被湮没

                    # If periodic particles, add a new particle at a random position at the same depth as the lost particle
                    # If not, particle is lost to the model
                    # 如果是周期性粒子，则在与丢失粒子相同深度的随机位置添加一个新粒子，如果没有，粒子就会消失在模型中
                    if params.periodic_particles: # 1
                        x_p = int(np.random.randint(low=0, high=grid_size))
                        y_p = int(np.random.randint(low=0, high=grid_size))
                        z_p = grid_new[x_p, y_p]
                    else:
                        x_p = np.nan
                        y_p = np.nan
                        z_p = np.nan
                    # FINAL PARTICLE POSITION DETERMINED - C2:S1

                    if plot_on:
                        plt.plot(x_p0, z_p0, 'kX')

                # 情况2：表面弹射 在瞬时撞击坑内
                elif 0.0 < R0 <= transient_crater_radius:
                    # SCENARIO 2: Surface ejection
                    if print_on:
                        print('CLASS 2 - SCENARIO 2')

                    # Interpolate on post-crater grid for computing landing position
                    xi = range(grid_size)
                    yi = range(grid_size)
                    XX, YY = np.meshgrid(xi, yi)
                    f_surf_new = interpolate.interp2d(xi, yi, grid_new, kind='linear') # 表面高程提取器
                    phi0 = np.arctan2(dy, dx) # 颗粒在撞击点的方向
                    ejection_velocity_vertical = alpha/(R0**3)  # 公式（8）垂直速度
                    # Initial guess for the time it will take the particle to land back on the surface
                    # Would be exact if no pre-existing topography
                    # 初步推测粒子返回地面所需的时间， 如果没有预先存在的地形，将是精确的
                    flight_time = 2.0*ejection_velocity_vertical/g # 垂直速度在重力加速度下变0的时间的二倍
                    # Solve for when and where the particle lands # 求解粒子着陆的时间和地点
                    res_solve = minimize(self.solve_for_landing_point, flight_time, args=(R0, ejection_velocity_vertical, x_crater_pix, y_crater_pix, phi0, f_surf_new, z_p0), bounds=[(0.0, np.inf)], method='L-BFGS-B')
                    # Flight time that minimizes the distance between the particle and the surface (aka finds when it lands on the landscape)
                    #  使粒子与地表之间的距离最小化的飞行时间（也就是当粒子落在地表上时的发现时间）
                    t_land = res_solve.x[0]

                    r_flight = R0 + ejection_velocity_vertical*t_land
                    z_flight = z_p0 + ejection_velocity_vertical*t_land - 0.5*g*(t_land**2)

                    R_land = np.hypot(r_flight, z_flight)
                    theta_land = np.arccos(z_flight/R_land)

                    if np.isnan(theta_land):
                        print('NAN LANDING THETA- SURFACE EJECTION')
                        sys.exit()
                    # 着陆点的位置
                    x_land = int(round(R_land*np.sin(theta_land)*np.cos(phi0)/resolution + x_crater_pix))
                    y_land = int(round(R_land*np.sin(theta_land)*np.sin(phi0)/resolution + y_crater_pix))

                    #  粒子落在网格上
                    if (0 <= x_land <= (grid_size-1)) and (0 <= y_land <= (grid_size-1)):
                        # Particle lands on the grid
                        x_p = int(x_land)
                        y_p = int(y_land)

                        dx_final = (x_p - x_crater_pix)*resolution
                        dy_final = (y_p - y_crater_pix)*resolution
                        dist_from_crater = np.hypot(dx_final, dy_final)
                        # 如果粒子落在连续的喷出岩毯中，则将其随机放置在落点处的喷出岩毯厚度范围内（即该点总高程变化的 1/3）
                        # If particle lands within the continuous ejecta blanket, place it randomly within the ejecta blanket thickness at its landing point (which is 1/3 the total elevation change at that point)
                        if dist_from_crater <= continuous_ejecta_blanket_factor*crater_radius:
                            z_p = grid_new[x_p, y_p] - np.random.rand()*(1.0/3.0)*abs(grid_new[x_p, y_p] - grid_old[x_p, y_p])
                        else:
                            z_p = grid_new[x_p, y_p]

                        if plot_on:
                            # For plotting purposes only
                            plt.plot(x_p0, z_p0, 'yX')

                            t_arr = np.linspace(0.0, t_land, 100)
                            r_t_flight = R0 + ejection_velocity_vertical*t_arr
                            z_t_flight = z_p0 + ejection_velocity_vertical*t_arr - 0.5*g*(t_arr**2)

                            R_t_flight = np.hypot(r_t_flight, z_t_flight)
                            theta_t_flight = np.arccos(z_t_flight/R_t_flight)

                            x_t_flight = R_t_flight*np.sin(theta_t_flight)*np.cos(phi0)/resolution + x_crater_pix
                            y_t_flight = R_t_flight*np.sin(theta_t_flight)*np.sin(phi0)/resolution + y_crater_pix

                            plt.plot(x_t_flight, z_t_flight, 'y--')
                            plt.plot(x_p, z_p, 'ro', markersize=3)
                    # 颗粒着陆在网格外
                    else:
                        # Particle lands off the grid
                        # 如果是周期性粒子，则在与丢失粒子相同深度的随机位置添加一个新粒子， 如果没有，粒子就会消失在模型中
                        # If periodic particles, add a new particle at a random position at the same depth as the lost particle
                        # If not, particle is lost to the model
                        if params.periodic_particles:
                            x_p = int(np.random.randint(low=0, high=grid_size))
                            y_p = int(np.random.randint(low=0, high=grid_size))
                            z_p = grid_new[x_p, y_p]
                        else:
                            x_p = np.nan
                            y_p = np.nan
                            z_p = np.nan

                        if plot_on:
                            plt.plot(x_p0, z_p0, 'cX')
                    # FINAL PARTICLE POSITION DETERMINED - C2:S2

                # 情景3：填充 在瞬时撞击坑外但是在撞击坑内
                elif transient_crater_radius < R0 <= crater_radius:
                    # SCENARIO 3: Infill
                    if print_on:
                        print('CLASS 2 - SCENARIO 3')
                    # Particle starts between the transient crater radius and final crater radius, ends up being part of the infilling material. Particle is placed randomly within the breccia lens at that distance
                    # 颗粒起始于瞬态火山口半径和最终火山口半径之间，最终成为填充材料的一部分。颗粒在该距离的角砾岩透镜内随机放置
                    if dz <= 0.0: # 如果在地上
                        theta0 = np.arccos(abs(dz)/R0)
                    elif dz > 0.0: # 如果在地下
                        theta0 = np.pi/2.0 + np.arcsin(dz/R0)
                    # 计算发射仰角
                    phi0 = np.arctan2(dy, dx)
                    # 计算发射方位角

                    R_final = np.random.uniform()*R0
                    # 计算着陆点
                    x_land = int(np.round(R_final*np.sin(theta0)*np.cos(phi0)/resolution + x_crater_pix))
                    y_land = int(np.round(R_final*np.sin(theta0)*np.sin(phi0)/resolution + y_crater_pix))

                    if (0 <= x_land <= (grid_size-1)) and (0 <= y_land <= (grid_size-1)):
                        # 粒子落在网格上
                        # Particle lands on the grid
                        x_p = int(x_land)
                        y_p = int(y_land)
                        z_p = grid_new[x_p, y_p] - np.random.uniform()*(1.0/2.0)*((3.0/8.0)*(2.0*crater_radius/1.17)*(1.0 - (R_final/crater_radius)))



                        if plot_on:
                            # For plotting purposes only
                            plt.plot(x_p0, z_p0, 'go', markersize=3)
                            plt.plot(x_p, z_p, 'ro', markersize=3)
                            plt.plot([x_p0, x_p], [z_p0, z_p], 'b--')
                    else:
                        # 粒子落在网格外
                        # Particle lands off the grid

                        # If periodic particles, add a new particle at a random position at the same depth as the lost particle
                        # If not, particle is lost to the model
                        if params.periodic_particles:
                            x_p = int(np.random.randint(low=0, high=grid_size))
                            y_p = int(np.random.randint(low=0, high=grid_size))
                            z_p = grid_new[x_p, y_p]
                        else:
                            x_p = np.nan
                            y_p = np.nan
                            z_p = np.nan

                        if plot_on:
                            # For plotting purposes only
                            plt.plot(x_p0, z_p0, 'bX', markersize=3)
                    # FINAL PARTICLE POSITION DETERMINED - C2:S3

                # 情景 4：地表掩埋
                elif R0 > crater_radius:
                    # SCENARIO 4: Surface burial
                    if print_on:
                        print('CLASS 2 - SCENARIO 4')
                    # Particle starts outside the final crater radius, gets buried by continuous ejecta blanket
                    # 粒子从最终陨石坑半径外开始，被连续的喷出物毯掩埋

                    x_p = int(x_p0)
                    y_p = int(y_p0)
                    z_p = z_p0

                    if z_p > grid_new[x_p, y_p]:
                        # 如果陨石坑落在其他陨石坑之上，可能会产生相互作用，甚至在新陨石坑外部造成海拔的净降低（见：继承参数）
                        # Possible that interactions where craters land on top of other craters that could cause a net decrease in elevation even exterior the the new crater (see: Inheritance parameter)
                        z_p = grid_new[x_p, y_p]

                    if plot_on:
                        # 仅用于绘图
                        # For plotting purposes only
                        plt.plot(x_p0, z_p0, 'gX')
                    # FINAL PARTICLE POSITION DETERMINED - C2:S4

            # 第 3 类：在影响范围内埋藏在地下的粒子
            elif d_p0 > 0.0:
                ##### ------------------------------------------------------------------ #####
                # CLASS 3: Particle buried in the subsurface within the sphere of influence

                if R0 >= continuous_ejecta_blanket_factor*crater_radius:
                    # SCENARIO 0: Particle outside the sphere of influence. Leave it where it is
                    # I have extended the initial threshold radius to be passed to the function to catch edge cases so these might get thrown in
                    x_p = int(x_p0)
                    y_p = int(y_p0)
                    z_p = z_p0

                    # Possible that interactions where craters land on top of other craters that could cause a net decrease in elevation even exterior the the new crater (see: Inheritance parameter)
                    if z_p > grid_new[x_p, y_p]:
                        z_p = grid_new[x_p, y_p]
                    # FINAL PARTICLE POSITION DETERMINED - C3:S0

                elif dx == 0.0 and dy == 0.0:
                    # SCENARIO 1: Drilling
                    if print_on:
                        print('CLASS 3 - SCENARIO 1')
                    # Only need to compute R(t) because the particle will not change x or y position. Theta0 = 0, phi0 = 0
                    theta0 = 0.0
                    phi0 = 0.0

                    R_flow = -1.0*((R0**4 + 4.0*alpha*t_flow)**(0.25))

                    x_p = int(x_p0)
                    y_p = int(y_p0)
                    z_p = R_flow

                    if z_p > grid_new[x_p, y_p]:
                        z_p = grid_new[x_p, y_p]
                    # FINAL PARTICLE POSITION DETERMINED - C3:S1

                    if plot_on:
                        t_arr = np.linspace(0.0, t_flow, 100)
                        R_flow = -1.0*((R0**4 + 4.0*alpha*t_arr)**(0.25))

                        x_flow = x_p0*np.ones(len(R_flow))
                        y_flow = y_p0*np.ones(len(R_flow))
                        z_flow = R_flow

                        plt.plot(x_p0, z_p0, 'go', markersize=3)
                        plt.plot(x_flow, z_flow, 'r--')
                        plt.plot(x_p, z_p, 'ro', markersize=3)

                else:
                    # Particle will move along a streamline that changes its [x,y,z] position and possibly eject it from the subsurface

                    if dz <= 0.0:
                        theta0 = np.arccos(abs(dz)/R0)
                    elif dz > 0.0:
                        theta0 = np.pi/2.0 + np.arcsin(dz/R0)
                    phi0 = np.arctan2(dy, dx)

                    # Interpolate on pre- and post-crater grid for computing ejection/landing positions
                    xi = range(grid_size)
                    yi = range(grid_size)
                    XX, YY = np.meshgrid(xi, yi)
                    f_surf_old = interpolate.interp2d(xi, yi, grid_old, kind='linear')

                    # Initial guess for ejection time.  Guess it will be ejected at an angle of ~90 degrees from the vertical
                    theta_eject_guess = np.pi/2.0
                    t_eject_guess = (R0**4)/(4.0*alpha)*(((1.0 - np.cos(theta0))**(-4)) - 1.0)
                    t_max = (R0**4)/(4.0*alpha)*((16.0*(1.0 - np.cos(theta0))**(-4)) - 1.0)

                    R_eject_guess = (R0**4 + 4.0*alpha*t_eject_guess)**(0.25)
                    theta_eject_guess = np.arccos(1.0 - ((1.0-np.cos(theta0))*(R_eject_guess/R0)))

                    # Solve for ejection time
                    res_solve = minimize(self.solve_for_ejection_point, t_eject_guess, args=(R0, x_crater_pix, y_crater_pix, phi0, theta0, alpha, f_surf_old), bounds=[(0.0, 0.99*t_max)], method='L-BFGS-B')
                    # Ejection time that minimizes the distance between the particle and the surface (i.e. finds when the streamline intersects the surface)
                    t_eject = res_solve.x[0]
                    R_eject = (R0**4 + 4.0*alpha*t_eject)**(0.25)

                    if t_eject <= t_flow and R_eject <= transient_crater_radius:
                        # SCENARIO 2: Subsurface ejection
                        if print_on:
                            print('CLASS 3 - SCENARIO 2')
                        # Particle streamline reaches the surface before the flow freezes and at a radial distance of less than the transient crater radius
                        f_surf_new = interpolate.interp2d(xi, yi, grid_new, kind='linear')

                        theta_eject = np.arccos(1.0 - ((1.0-np.cos(theta0))*(R_eject/R0)))

                        x_eject = (R_eject*np.sin(theta_eject)*np.cos(phi0))/resolution + x_crater_pix
                        y_eject = (R_eject*np.sin(theta_eject)*np.sin(phi0))/resolution + y_crater_pix
                        z_eject = -1.0*R_eject*np.cos(theta_eject)

                        ejection_velocity_vertical = alpha/(R_eject**3)

                        # Initial guess for the time it will take the particle to land back on the surface
                        # Would be exact if no pre-existing topography
                        flight_time = 2.0*ejection_velocity_vertical/g

                        # Solve for when and where the particle lands
                        res_solve = minimize(self.solve_for_landing_point, flight_time, args=(R_eject, ejection_velocity_vertical, x_crater_pix, y_crater_pix, phi0, f_surf_new, z_eject), bounds=[(0.0, np.inf)], method='L-BFGS-B')
                        # Flight time that minimizes the distance between the particle and the surface (aka finds when it lands on the landscape)
                        t_land = res_solve.x[0]

                        r_flight = R_eject + ejection_velocity_vertical*t_land
                        z_flight = z_eject + ejection_velocity_vertical*t_land - 0.5*g*(t_land**2)

                        R_land = np.hypot(r_flight, z_flight)
                        theta_land = np.arccos(z_flight/R_land)

                        if np.isnan(theta_land):
                            print('NAN LANDING THETA- SUBSURFACE EJECTION')
                            sys.exit()

                        x_land = int(round(R_land*np.sin(theta_land)*np.cos(phi0)/resolution + x_crater_pix))
                        y_land = int(round(R_land*np.sin(theta_land)*np.sin(phi0)/resolution + y_crater_pix))

                        if (0 <= x_land <= (grid_size-1)) and (0 <= y_land <= (grid_size-1)):
                            # Particle lands on the grid
                            x_p = int(x_land)
                            y_p = int(y_land)

                            dx_final = (x_p - x_crater_pix)*resolution
                            dy_final = (y_p - y_crater_pix)*resolution

                            dist_from_crater = np.hypot(dx_final, dy_final)

                            # If particle lands within the continuous ejecta blanket, place it randomly within the ejecta blanket thickness at its landing point (which is 1/3 the total elevation change)
                            if dist_from_crater < continuous_ejecta_blanket_factor*crater_radius:
                                z_p = grid_new[x_p, y_p] - np.random.rand()*(1.0/3.0)*abs(grid_new[x_p, y_p] - grid_old[x_p, y_p])
                            else:
                                z_p = grid_new[x_p, y_p]

                            if plot_on:
                                # For plotting purposes only
                                t_arr = np.linspace(0.0, t_eject, 100)
                                R_flow = (R0**4 + 4.0*alpha*t_arr)**(0.25)

                                theta_flow = np.arccos(1.0 - ((1.0-np.cos(theta0))*(R_flow/R0)))

                                x_flow = (R_flow*np.sin(theta_flow)*np.cos(phi0))/resolution + x_crater_pix
                                y_flow = (R_flow*np.sin(theta_flow)*np.sin(phi0))/resolution + y_crater_pix
                                z_flow = -1.0*R_flow*np.cos(theta_flow)

                                plt.plot(x_p0, z_p0, 'go', markersize=3)
                                plt.plot(x_flow, z_flow, 'r--')
                                plt.plot(x_flow[-1], z_flow[-1], 'yX')

                                t_arr = np.linspace(0.0, t_land, 100)

                                r_flight = R_eject + ejection_velocity_vertical*t_arr
                                z_flight = z_eject + ejection_velocity_vertical*t_arr - 0.5*g*(t_arr**2)

                                R_flight = np.hypot(r_flight, z_flight)
                                theta_flight = np.arccos(z_flight/R_flight)

                                x_flight = R_flight*np.sin(theta_flight)*np.cos(phi0)/resolution + x_crater_pix
                                y_flight = R_flight*np.sin(theta_flight)*np.sin(phi0)/resolution + y_crater_pix

                                plt.plot(x_flight, z_flight, 'y--')
                                plt.plot(x_p, z_p, 'ro', markersize=3)

                        else:
                            # Particle lands off the grid

                            # If periodic particles, add a new particle at a random position at the same depth as the lost particle
                            # If not, particle is lost to the model
                            if params.periodic_particles:
                                x_p = int(np.random.randint(low=0,high=grid_size))
                                y_p = int(np.random.randint(low=0,high=grid_size))
                                z_p = grid_new[x_p, y_p]
                            else:
                                x_p = np.nan
                                y_p = np.nan
                                z_p = np.nan

                            if plot_on:

                                t_arr = np.linspace(0.0, t_eject, 100)
                                R_flow = (R0**4 + 4.0*alpha*t_arr)**(0.25)

                                theta_flow = np.arccos(1.0 - ((1.0-np.cos(theta0))*(R_flow/R0)))

                                x_flow = (R_flow*np.sin(theta_flow)*np.cos(phi0))/resolution + x_crater_pix
                                y_flow = (R_flow*np.sin(theta_flow)*np.sin(phi0))/resolution + y_crater_pix
                                z_flow = -1.0*R_flow*np.cos(theta_flow)

                                plt.plot(x_p0, z_p0, 'go', markersize=3)
                                plt.plot(x_flow, z_flow, 'r--')
                                plt.plot(x_flow[-1], z_flow[-1], 'cX')
                        # FINAL PARTICLE POSITION DETERMINED - C3:S2

                    elif t_eject <= t_flow:
                        # SCENARIO 3: Unphysical subsurface transport to above the surface (constant alpha)
                        if print_on:
                            print('CLASS 3 - SCENARIO 3A')

                        # Particle streamline reaches the surface before the flow freezes and at a radial distance of less than the transient crater radius
                        f_surf_new = interpolate.interp2d(xi, yi, grid_new, kind='linear')

                        theta_eject = np.arccos(1.0 - ((1.0-np.cos(theta0))*(R_eject/R0)))

                        x_eject = int(round((R_eject*np.sin(theta_eject)*np.cos(phi0))/resolution + x_crater_pix))
                        y_eject = int(round((R_eject*np.sin(theta_eject)*np.sin(phi0))/resolution + y_crater_pix))
                        z_eject = -1.0*R_eject*np.cos(theta_eject)

                        if (0 <= x_eject <= (grid_size-1)) and (0 <= y_eject <= (grid_size-1)):
                            x_p = int(x_eject)
                            y_p = int(y_eject)
                            z_p = z_eject

                            if z_p > grid_new[x_p, y_p]:
                                z_p = grid_new[x_p, y_p]
                                '''
                                print('PARTICLE ABOVE SURFACE - NEW SHIT')
                                print(x_eject, y_eject, z_eject)
                                print(x_p, y_p, z_p)
                                print(grid_old[x_eject, y_eject], grid_new[x_eject, y_eject])
                                plt.figure()
                                plt.subplot(211)
                                plt.imshow(grid_old.T)
                                plt.scatter(x_p0, y_p0, c='g', s=2)
                                plt.scatter(x_crater_pix, y_crater_pix, c='b', s=2)
                                plt.scatter(x_eject, y_eject, c='r', s=2)
                                plt.subplot(212)
                                plt.imshow(grid_new.T)
                                plt.scatter(x_p0, y_p0, c='g', s=2)
                                plt.scatter(x_crater_pix, y_crater_pix, c='b', s=2)
                                plt.scatter(x_eject, y_eject, c='r', s=2)
                                plt.show()
                                sys.exit()
                                '''

                        else:
                            if params.periodic_particles:
                                x_p = int(np.random.randint(low=0, high=grid_size))
                                y_p = int(np.random.randint(low=0, high=grid_size))
                                # Assume that the surface outside the grid is zero everywhere.
                                z_p = grid_new[x_p, y_p]

                            else:
                                x_p = np.nan
                                y_p = np.nan
                                z_p = np.nan

                        # Particle moves along a streamline that does not reach the surface before the flow freezes. Or it does reach the surface but
                        # does so at a radial distance greater than the transient crater radius.  Particles at these distances are not observed to be ejected.
                        # This artifact is likely due to our use of a constant alpha, meaning that the velocity field does not decay with time.

                        # FINAL PARTICLE POSITION DETERMINED - C3:S3A

                    else:
                        # Normal subsurface transport
                        if print_on:
                            print('CLASS 3 - SCENARIO 3B')

                        R_flow = (R0**4 + 4.0*alpha*t_flow)**(0.25)
                        theta_arg = 1.0 - ((1.0-np.cos(theta0))*(R_flow/R0))

                        try:
                            theta_flow = np.arccos(theta_arg)
                        except:
                            print('NAN SUBSURFACE FLOW THETA')
                            print(x_p0, y_p0, z_p0)
                            print(x_crater_pix, y_crater_pix)
                            print(dx, dy, dz)
                            print(R0, crater_radius)
                            print(theta0, R_flow)
                            print(1.0 - ((1.0-np.cos(theta0))*(R_flow/R0)))
                            sys.exit()

                        x_flow = int(round((R_flow*np.sin(theta_flow)*np.cos(phi0))/resolution + x_crater_pix))
                        y_flow = int(round((R_flow*np.sin(theta_flow)*np.sin(phi0))/resolution + y_crater_pix))
                        z_flow = -1.0*R_flow*np.cos(theta_flow)

                        if (0 <= x_flow <= (grid_size-1)) and (0 <= y_flow <= (grid_size-1)):
                            # Particle flows on the grid
                            x_p = int(x_flow)
                            y_p = int(y_flow)
                            z_p = z_flow
                            # Super-surface flows - NEED TO FIX.  For now just place on the surface at the current position
                            if z_p > grid_new[x_p, y_p]:
                                z_p = grid_new[x_p, y_p]

                            if plot_on:
                                t_arr = np.linspace(0.0, t_flow, 100, dtype=np.float)
                                R_flow = (R0**4 + 4.0*alpha*t_arr)**(0.25)

                                theta_flow = np.arccos(1.0 - ((1.0-np.cos(theta0))*(R_flow/R0)))

                                x_flow = (R_flow*np.sin(theta_flow)*np.cos(phi0))/resolution + x_crater_pix
                                y_flow = (R_flow*np.sin(theta_flow)*np.sin(phi0))/resolution + y_crater_pix
                                z_flow = -1.0*R_flow*np.cos(theta_flow)

                                plt.plot(x_p0, z_p0, 'go', markersize=3)
                                plt.plot(x_flow, z_flow, 'r--')
                                plt.plot(x_p, z_p, 'ro', markersize=3)
                        else:
                            # Particle flows off the grid

                            # If periodic particles, add a new particle at a random position at the same depth as the lost particle
                            # If not, particle is lost to the model
                            if params.periodic_particles:
                                x_p = int(np.random.randint(low=0, high=grid_size))
                                y_p = int(np.random.randint(low=0, high=grid_size))
                                # Assume that the surface outside the grid is zero everywhere.
                                z_p = grid_new[x_p, y_p] - abs(z_flow)
                                # Particle would then be buried at a depth of z_flow so place it randomly on the grid at that depth
                            else:
                                x_p = np.nan
                                y_p = np.nan
                                z_p = np.nan

                            if plot_on:
                                plt.plot(x_p0, z_p0, 'rX')
                        # FINAL PARTICLE POSITION DETERMINED - C3:S3

        d_p = grid_new[int(x_p), int(y_p)] - z_p
        if d_p < 0.0:
            print('PARTICLE ABOVE THE SURFACE - END OF CRATER FUNCTION')
            sys.exit()
            z_p = grid_new[x_p, y_p]

        return [x_p, y_p, z_p]

    def sample_noise_val(self):
        # Sample elevation noise values from a given distribution
        dist = st.johnsonsu

        params = [0.7694228050938363, 1.0370825784482083, 0.14462393568186127, 0.11525690456701247]

        noise_val = dist.rvs(params[0], params[1], params[2], size=1)[0]

        return noise_val

    def tracer_particle_noise(self, x_p0, y_p0, z_p0, grid_old, noise):
        # Movement of tracer particles from the addition of sub-pixel cratering
        # Particles are not moved horizontally.  If cratering removes material from the particle's pixel and excavates the particle to be above the surface,
        # Place the particle on the new surface at the same pixel

        # Change particle z-position
        if noise == 0.0:
            z_p_new = z_p0

        elif noise < 0.0:
            z_p_new = z_p0 + abs(noise)

        elif noise > 0.0:
            z_p_new = z_p0 - abs(noise)

        # Check if particle has been unearthed above the pixel reference elevation
        if z_p_new > grid_old[x_p0, y_p0]:
            x_p = int(x_p0)
            y_p = int(y_p0)
            z_p = grid_old[x_p, y_p]

        else:
            x_p = int(x_p0)
            y_p = int(y_p0)
            z_p = z_p_new

        d_p = grid_old[int(x_p), int(y_p)] - z_p
        if d_p < 0.0:
            print('PARTICLE ABOVE THE SURFACE - END OF NOISE FUNCTION')
            sys.exit()
            z_p = grid_new[x_p, y_p]

        return [x_p, y_p, z_p]




def build_tracers_group():

    n_pd = params.n_particles_per_layer  # 10
    x_points = np.linspace(5, params.grid_size - 5, n_pd, dtype=np.int32)
    y_points = np.linspace(5, params.grid_size - 5, n_pd, dtype=np.int32)
    z_points = -1.0 * np.linspace(0, 1, 100)
    z_points[0] = 0.0
    print(z_points)


    XX, YY, ZZ = np.meshgrid(x_points, y_points, z_points)
    # shape (25, 25, 25) (25, 25, 25) (25, 25, 25)
    tracers = [] # 存储粒子的表 x, y, z 共25*25*25个粒子
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            for k in range(XX.shape[2]):
                x_p0 = XX[i, j, k]
                y_p0 = YY[i, j, k]
                z_p0 = ZZ[i, j, k]

                tracer = Tracer(x_p0, y_p0, z_p0)
                tracers.append(tracer)
    return tracers
