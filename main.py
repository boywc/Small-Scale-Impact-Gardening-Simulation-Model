import params
import numpy as np
from Grid import Grid
from Tracers import build_tracers_group
from ImpactorPopulation import ImpactorPopulation, save_craters
import matplotlib.pyplot as plt


if __name__ == '__main__':

    impPop = ImpactorPopulation()
    d_craters, x_craters, y_craters, index_craters, t_craters = impPop.sample_all_craters()
    save_craters(d_craters, x_craters, y_craters, index_craters, t_craters)
    tracers = build_tracers_group()

    grid = Grid(params.grid_size, params.resolution, params.diffusivity, params.dt)
    grid_old = grid.setUpGrid()
    grid_new = np.copy(grid_old)

    t_line = []
    partical_depth_all = []
    for t in range(params.nsteps):
        print("*******************************************************************")
        print("Time step: %d / %d"%(t, params.nsteps))
        timestep_index = np.where(t_craters == t)
        current_diameters = d_craters[timestep_index]
        current_x = x_craters[timestep_index]
        current_y = y_craters[timestep_index]
        current_index = index_craters[timestep_index]
        index_shuf = list(range(len(current_diameters)))
        np.random.shuffle(index_shuf)
        current_diameters = np.array([current_diameters[j] for j in index_shuf])
        current_x = np.array([current_x[j] for j in index_shuf])
        current_y = np.array([current_y[j] for j in index_shuf])
        current_index = np.array([current_index[j] for j in index_shuf])
        t_start = t * params.dt
        imp_dt = params.dt / len(current_diameters)
        for i in range(len(current_diameters)):
            imp_t = t_start + imp_dt * i
            t_line.append(imp_t)
            print("\r\tCalculate Crater: %d / %d " % (i, len(current_diameters)), end="")
            crater_diam = current_diameters[i]
            crater_radius = crater_diam / 2.0
            crater_index = current_index[i]

            x_crater_pix = int(current_x[i])
            y_crater_pix = int(current_y[i])
            ones_grid = np.ones((params.grid_size, params.grid_size))
            X_grid, Y_grid = np.ogrid[:params.grid_size, :params.grid_size]
            # X_grid (500, 1)  [[0], ... [499]]
            # Y_grid (1, 500)  [[0,  ...  499]]

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
            # np.copyto(grid_old, grid_new)

            partical_depth_now = []
            for j in range(len(tracers)):
                particle_position = tracers[j].current_position()
                x_p0 = particle_position[0]
                y_p0 = particle_position[1]
                z_p0 = particle_position[2]
                if ~np.isnan(x_p0) and ~np.isnan(y_p0) and ~np.isnan(z_p0):
                    x_p0 = int(x_p0)
                    y_p0 = int(y_p0)
                    z_p0 = z_p0
                    d_p0 = grid_old[x_p0, y_p0] - z_p0
                    if d_p0 < 0.0: # 如果颗粒飞腾，让他落回表面
                        z_p0 = grid_old[x_p0, y_p0]
                        d_p0 = grid_old[x_p0, y_p0] - z_p0
                    partical_depth_now.append(d_p0)
                    dx = (x_p0 - x_crater_pix) * params.resolution
                    dy = (y_p0 - y_crater_pix) * params.resolution
                    if (0 <= x_crater_pix <= (params.grid_size - 1)) and (0 <= y_crater_pix <= (params.grid_size - 1)):
                        dz = z_p0 - grid_old[x_crater_pix, y_crater_pix]
                    else:
                        dz = z_p0
                    R0 = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    if R0 <= 5.0 * params.continuous_ejecta_blanket_factor * crater_radius:
                        particle_position_new = tracers[j].tracer_particle_crater(x_p0, y_p0, z_p0, d_p0, dx, dy, dz,
                                                                                  x_crater_pix, y_crater_pix, R0,
                                                                                  crater_radius, grid_old, grid_new)

                        tracers[j].update_position(particle_position_new)
                else:
                    partical_depth_now.append(-9999)

            partical_depth_all.append(partical_depth_now)
            np.copyto(grid_old, grid_new)

        print("\n")

    t_line = np.array(t_line)
    partical_depth_all = np.array(partical_depth_all)
    np.save(params.save_dir + 't_line.npy', t_line)
    np.save(params.save_dir + 'partical_depth.npy', partical_depth_all)
    np.save(params.save_dir + 'Crater_Map.npy', grid_new)
    plt.imshow(grid_new)
    plt.show()
